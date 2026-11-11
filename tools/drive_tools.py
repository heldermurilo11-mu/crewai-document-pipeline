import os
import io
from typing import List, Set, Optional, Dict, Any, TYPE_CHECKING
from crewai.tools import BaseTool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class GoogleDriveDownloadTool(BaseTool):
    name: str = "Google Drive Downloader"
    description: str = (
        "Conecta ao Google Drive, lista arquivos de uma pasta específica "
        "e baixa eles para processamento local."
    )
    
    credentials_dict: Optional[dict] = None
    token_dict: Optional[dict] = None

    def _get_service(self):
        """Gerencia a autenticação OAuth2 do Google."""
        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = None
        
        # 1. Tenta carregar do Token injetado (Vault)
        if self.token_dict:
            try:
                creds = Credentials.from_authorized_user_info(self.token_dict, SCOPES)
                logger.info("🔑 Usando Token do Vault.")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao carregar Token do Vault: {e}")

        # 2. Se não conseguiu, tenta do arquivo token.json local
        token_path = os.path.join(BASE_DIR, 'token.json')
        if not creds and os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
                logger.info("🔑 Usando token.json local.")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao carregar token.json local: {e}")
            
        # 3. Validação e Refresh
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    logger.info("🔄 Token atualizado via Refresh Token.")
                except Exception as e:
                    logger.error(f"❌ Falha no refresh do token: {e}")
                    creds = None

            # 4. Se ainda não tem creds válidas, tenta iniciar fluxo de auth
            if not creds:
                credentials_path = os.path.join(BASE_DIR, 'credentials.json')
                if self.credentials_dict:
                    logger.info("🔐 Iniciando fluxo de auth com Credentials do Vault.")
                    flow = InstalledAppFlow.from_client_config(
                        self.credentials_dict, SCOPES)
                elif os.path.exists(credentials_path):
                    logger.info("🔐 Iniciando fluxo de auth com credentials.json local.")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        credentials_path, SCOPES)
                else:
                    raise FileNotFoundError("Nenhuma credencial encontrada (Vault ou Local)!")
                
                # Este passo exige interação humana (browser)
                creds = flow.run_local_server(port=0)
            
            # Salva o novo token localmente para facilitar dev futuro (opcional)
            with open(os.path.join(BASE_DIR, 'token.json'), 'w') as token:
                token.write(creds.to_json())

        return build('drive', 'v3', credentials=creds)

    def list_files(self, folder_id: str) -> List[Dict[str, str]]:
        """Lista todos os arquivos de uma pasta com suporte a paginação.

        Args:
            folder_id: ID da pasta no Google Drive.

        Returns:
            Lista de dicionários contendo {'id': ..., 'name': ...}.
        """
        service = self._get_service()
        all_items = []
        next_page_token = None
        query = f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"

        while True:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name)",
                pageToken=next_page_token
            ).execute()

            items = results.get('files', [])
            all_items.extend(items)

            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break

        # Limpa nomes (strip)
        for item in all_items:
            item['name'] = item['name'].strip()

        return all_items

    def download_file(self, file_id: str, file_name: str) -> Optional[str]:
        """Baixa um único arquivo do Google Drive por ID.

        Args:
            file_id: ID do arquivo no Drive.
            file_name: Nome que o arquivo terá localmente.

        Returns:
            Caminho absoluto do arquivo baixado ou None se falhar.
        """
        service = self._get_service()
        local_folder = os.path.join(BASE_DIR, "input_files")
        os.makedirs(local_folder, exist_ok=True)
        file_path = os.path.join(local_folder, file_name)

        try:
            logger.info(f"📥 Baixando arquivo: {file_name}...")
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            logger.info(f"✅ Download concluído: {file_name}")
            return file_path
        except Exception as e:
            logger.error(f"❌ Erro ao baixar {file_name}: {e}")
            return None

    def get_or_create_subfolder(self, parent_folder_id: str, subfolder_name: str) -> str:
        """Busca uma subpasta pelo nome dentro de uma pasta pai. Cria se não existir.

        Args:
            parent_folder_id: ID da pasta pai no Drive.
            subfolder_name: Nome da subpasta.

        Returns:
            ID da subpasta encontrada ou criada.
        """
        service = self._get_service()
        query = (
            f"'{parent_folder_id}' in parents and "
            f"name = '{subfolder_name}' and "
            f"mimeType = 'application/vnd.google-apps.folder' and "
            f"trashed = false"
        )
        results = service.files().list(q=query, fields="files(id)").execute()
        folders = results.get('files', [])

        if folders:
            logger.info(f"📁 Subpasta '{subfolder_name}' encontrada.")
            return folders[0]['id']

        # Cria a subpasta
        metadata = {
            'name': subfolder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        folder = service.files().create(body=metadata, fields='id').execute()
        logger.info(f"📁 Subpasta '{subfolder_name}' criada.")
        return folder['id']

    def move_to_folder(self, file_id: str, destination_folder_id: str, file_name: str = "") -> bool:
        """Move um arquivo para outra pasta no Google Drive.

        Args:
            file_id: ID do arquivo a mover.
            destination_folder_id: ID da pasta de destino.
            file_name: Nome do arquivo (para logging).

        Returns:
            True se moveu com sucesso, False se falhou.
        """
        service = self._get_service()
        try:
            # Busca os pais atuais do arquivo
            file_info = service.files().get(fileId=file_id, fields='parents').execute()
            previous_parents = ','.join(file_info.get('parents', []))

            # Move: adiciona novo pai e remove o anterior
            service.files().update(
                fileId=file_id,
                addParents=destination_folder_id,
                removeParents=previous_parents,
                fields='id, parents'
            ).execute()

            logger.info(f"📂 Arquivo movido para processados: {file_name}")
            return True
        except Exception as e:
            logger.error(f"⚠️ Erro ao mover {file_name}: {e}")
            return False

    def _run(self, folder_id: str) -> List[Dict[str, str]]:
        """Método de compatibilidade para o CrewAI. Agora retorna metadados."""
        return self.list_files(folder_id)
