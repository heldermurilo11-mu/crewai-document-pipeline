"""Hardened Google Drive tool with OAuth2/SA auth, clamped deadlines, and atomic downloads."""

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import httplib2

from runtime.deadline import Deadline

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SCOPES = ["https://www.googleapis.com/auth/drive"]


def _query_literal(value: str) -> str:
    """Escape backslashes and single quotes for Google Drive search queries."""
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


@dataclass
class GoogleDriveDownloadTool:
    """Lists, downloads atomically, and idempotently moves PDF/image documents in Google Drive."""

    name: str = "Google Drive Downloader"
    description: str = (
        "Lista e baixa PDFs ou imagens do Google Drive com transporte com timeout, "
        "download atômico e movimentação idempotente de pastas."
    )
    credentials_dict: Optional[dict] = None
    token_dict: Optional[dict] = None
    deadline: Optional[Deadline] = None
    request_timeout: float = 120.0
    _service: Any = field(default=None, init=False, repr=False)
    _credentials: Any = field(default=None, init=False, repr=False)

    def _timeout(self) -> float:
        if self.deadline is not None:
            return self.deadline.timeout(self.request_timeout)
        return self.request_timeout

    def _get_service(self):
        if self._service is not None:
            self._service._http.http.timeout = self._timeout()
            return self._service

        scopes = list(SCOPES)
        creds = None
        token_info = self.token_dict
        token_path = BASE_DIR / "token.json"
        creds_path = BASE_DIR / "credentials.json"

        if not token_info and token_path.is_file():
            token_info = json.loads(token_path.read_text(encoding="utf-8"))

        if token_info:
            granted = token_info.get("scopes") or token_info.get("scope")
            if isinstance(granted, str):
                granted = granted.split()
            if granted:
                matched = [s for s in scopes if s in granted]
                if not matched:
                    raise RuntimeError("OAuth token grants none of the required Google Drive scopes")
                scopes = matched
            creds = Credentials.from_authorized_user_info(token_info, scopes)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(
                lambda *args, **kwargs: Request()(*args, **{**kwargs, "timeout": self._timeout()})
            )

        if not creds or not creds.valid:
            client_cfg = self.credentials_dict
            if not client_cfg and creds_path.is_file():
                client_cfg = json.loads(creds_path.read_text(encoding="utf-8"))

            if client_cfg:
                if client_cfg.get("type") == "service_account":
                    creds = service_account.Credentials.from_service_account_info(
                        client_cfg, scopes=scopes
                    )
                elif os.getenv("ALLOW_INTERACTIVE_AUTH", "true").lower() == "true":
                    flow = InstalledAppFlow.from_client_config(client_cfg, scopes)
                    creds = flow.run_local_server(port=0, timeout_seconds=int(self._timeout()))
                    token_path.write_text(creds.to_json(), encoding="utf-8")
                else:
                    raise RuntimeError("Interactive OAuth is disabled for unattended pipeline runs")
            else:
                creds, _ = google.auth.default(scopes=scopes)

        self._credentials = creds
        transport = AuthorizedHttp(
            creds,
            http=httplib2.Http(timeout=self._timeout()),
            max_refresh_attempts=1,
        )
        self._service = build(
            "drive",
            "v3",
            http=transport,
            cache_discovery=False,
            static_discovery=True,
        )
        return self._service

    def _execute(self, request):
        service = self._get_service()
        service._http.http.timeout = self._timeout()
        return request.execute(num_retries=0)

    def list_files(self, folder_id: str, limit: Optional[int] = None) -> list[dict]:
        """List PDFs and scanned images in *folder_id* with checksum/revision metadata."""
        service = self._get_service()
        safe_folder = _query_literal(folder_id)
        query = (
            f"'{safe_folder}' in parents and trashed = false "
            "and (mimeType = 'application/pdf' or mimeType contains 'image/')"
        )
        items: list[dict] = []
        page_token: Optional[str] = None
        while True:
            result = self._execute(
                service.files().list(
                    q=query,
                    pageSize=100,
                    pageToken=page_token,
                    orderBy="createdTime",
                    fields="nextPageToken,files(id,name,md5Checksum,modifiedTime,createdTime,size,mimeType)",
                )
            )
            items.extend(result.get("files", []))
            if limit is not None and len(items) >= limit:
                items = items[:limit]
                break
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        for item in items:
            item["name"] = str(item.get("name", "")).strip()
        return items

    def find_file_by_name(self, folder_id: str, file_name: str) -> Optional[dict]:
        service = self._get_service()
        query = (
            f"'{_query_literal(folder_id)}' in parents "
            f"and name = '{_query_literal(file_name)}' and trashed = false"
        )
        result = self._execute(service.files().list(q=query, fields="files(id,name,md5Checksum,modifiedTime)"))
        return next(iter(result.get("files", [])), None)

    def download_file(
        self,
        file_id: str,
        file_name: str,
        destination: Optional[str | Path] = None,
    ) -> Optional[str]:
        """Atomically stream *file_id* into `.partial`, fsync, and os.replace."""
        service = self._get_service()
        target = (
            Path(destination).resolve()
            if destination is not None
            else (BASE_DIR / "input_files" / Path(file_name).name).resolve()
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            with partial.open("wb") as stream:
                downloader = MediaIoBaseDownload(
                    stream,
                    service.files().get_media(fileId=file_id),
                    chunksize=4 * 1024 * 1024,
                )
                done = False
                while not done:
                    if self.deadline is not None:
                        self.deadline.check("download chunk")
                    service._http.http.timeout = self._timeout()
                    _, done = downloader.next_chunk(num_retries=0)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(partial, target)
            return str(target)
        except Exception as exc:
            logger.error("Erro no download atômico do arquivo %s (%s): %s", file_name, file_id, exc)
            raise
        finally:
            if partial.exists():
                partial.unlink(missing_ok=True)

    def find_subfolder_id(self, parent_folder_id: str, subfolder_name: str) -> Optional[str]:
        service = self._get_service()
        query = (
            f"'{_query_literal(parent_folder_id)}' in parents "
            f"and name = '{_query_literal(subfolder_name)}' "
            "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        result = self._execute(service.files().list(q=query, fields="files(id)"))
        return next((item["id"] for item in result.get("files", [])), None)

    def get_or_create_subfolder(self, parent_folder_id: str, subfolder_name: str) -> str:
        existing = self.find_subfolder_id(parent_folder_id, subfolder_name)
        if existing:
            return existing
        service = self._get_service()
        created = self._execute(
            service.files().create(
                body={
                    "name": subfolder_name,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": [parent_folder_id],
                },
                fields="id",
            )
        )
        return created["id"]

    def move_to_folder(self, file_id: str, destination_folder_id: str, file_name: str = "") -> bool:
        """Idempotently move *file_id* into *destination_folder_id*."""
        service = self._get_service()
        parents = self._execute(service.files().get(fileId=file_id, fields="parents")).get("parents", [])
        if parents == [destination_folder_id]:
            return True
        arguments: dict[str, Any] = {"fileId": file_id, "fields": "id,parents"}
        if destination_folder_id not in parents:
            arguments["addParents"] = destination_folder_id
        previous = [parent for parent in parents if parent != destination_folder_id]
        if previous:
            arguments["removeParents"] = ",".join(previous)
        result = self._execute(service.files().update(**arguments))
        moved = result.get("parents") == [destination_folder_id]
        if moved:
            logger.info("Arquivo movido no Drive: %s -> %s", file_name or file_id, destination_folder_id)
        return moved

    def _run(self, folder_id: str) -> list[dict]:
        return self.list_files(folder_id)
