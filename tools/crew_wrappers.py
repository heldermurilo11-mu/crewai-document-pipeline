"""CrewAI @tool adapters backed by hardened Drive, high-DPI OCR, Review, and atomic I/O."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from runtime.checkpoint import Checkpoint
from runtime.config import RuntimeConfig
from runtime.deadline import Deadline
from runtime.files import atomic_write_text
from ._crew_compat import tool
from .drive_tools import GoogleDriveDownloadTool
from .feedback_store import FeedbackStore
from .filler_agent import FillerAgentTool
from .pattern_identifier import PatternIdentifierTool
from .reviewer import Reviewer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

drive_tool = GoogleDriveDownloadTool()
identifier_tool = PatternIdentifierTool()
filler_tool = FillerAgentTool()
reviewer_tool = Reviewer()
feedback_tool = FeedbackStore()


def _format_csv_line(values: list[str]) -> str:
    return ";".join(str(v or "").replace(";", ",").replace("\n", " ").strip() for v in values)


@tool("Baixar_Arquivos_Drive")
def baixar_arquivos_drive(folder_id: str) -> str:
    """Lista e baixa PDFs e imagens de *folder_id* no Google Drive com download atômico."""
    logger.info("Listando arquivos no Google Drive folder=%s...", folder_id)
    config = RuntimeConfig.from_env()
    deadline = Deadline(config.batch_budget_seconds)
    drive_tool.deadline = deadline
    drive_tool.request_timeout = config.drive_request_timeout_seconds

    arquivos = drive_tool.list_files(folder_id=folder_id)
    if not arquivos:
        return json.dumps({"status": "no_files", "arquivos": [], "message": "Nenhum arquivo encontrado"})

    arquivos_baixados = []
    for arq in arquivos:
        fid = arq["id"]
        fname = arq["name"]
        path = drive_tool.download_file(file_id=fid, file_name=fname)
        if path:
            arquivos_baixados.append(
                {
                    "file_id": fid,
                    "file_name": fname,
                    "file_path": path,
                    "md5Checksum": arq.get("md5Checksum", ""),
                    "modifiedTime": arq.get("modifiedTime", ""),
                }
            )

    return json.dumps({"status": "success", "arquivos": arquivos_baixados}, ensure_ascii=False)


@tool("Extrair_Dados_Cadeia_Custodia")
def extrair_dados_cadeia_custodia(lista_arquivos_json: str) -> str:
    """Identifica o layout da COC e executa OCR 300 DPI com detecção de orientação e schema estrito."""
    try:
        dados = json.loads(lista_arquivos_json)
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "Entrada não é um JSON válido"})

    config = RuntimeConfig.from_env()
    deadline = Deadline(config.batch_budget_seconds)
    arquivos = dados if isinstance(dados, list) else dados.get("arquivos", [])
    resultados_totais = []

    for arq in arquivos:
        caminho = arq.get("file_path")
        nome = arq.get("file_name", Path(caminho).name if caminho else "desconhecido.pdf")
        fid = arq.get("file_id", "")

        if not caminho or not os.path.exists(caminho):
            resultados_totais.append({"file_name": nome, "file_id": fid, "error": "Arquivo local ausente"})
            continue

        ident = identifier_tool.identify(caminho, deadline=deadline)
        if "error" in ident:
            resultados_totais.append({"file_name": nome, "file_id": fid, "error": ident["error"]})
            continue

        work_dir = config.resolved_state_dir() / "crew_work" / (fid or Path(caminho).stem)
        filler_tool.runtime_config = config
        try:
            ext_data = filler_tool.extract(
                caminho,
                identification_result=ident,
                deadline=deadline,
                work_dir=work_dir,
            )
            ext_data["file_path"] = caminho
            ext_data["file_id"] = fid
            ext_data["file_name"] = nome
            ext_data["md5Checksum"] = arq.get("md5Checksum", "")
            ext_data["modifiedTime"] = arq.get("modifiedTime", "")
            resultados_totais.append(ext_data)
        except Exception as exc:
            logger.error("Erro na extração de %s: %s", nome, exc)
            resultados_totais.append(
                {
                    "file_name": nome,
                    "file_id": fid,
                    "file_path": caminho,
                    "template": ident.get("template_name", "GENERIC_TEMPLATE"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "samples": [],
                }
            )

    return json.dumps({"status": "success", "resultados": resultados_totais}, ensure_ascii=False)


@tool("Validar_e_Exportar_Dados")
def validar_e_exportar_dados(resultados_extracao_json: str) -> str:
    """Valida amostras extraídas, persiste checkpoint SQLite WAL e grava resultado_crew.csv atomicamente."""
    try:
        dados = json.loads(resultados_extracao_json)
    except Exception as exc:
        logger.error("Erro no json.loads da validação: %s", exc)
        return "Erro: Entrada JSON inválida."

    if isinstance(dados, list):
        resultados = dados
    elif isinstance(dados, dict):
        resultados = dados.get("resultados", [])
    else:
        resultados = []

    config = RuntimeConfig.from_env()
    csv_path = config.resolved_output_csv()
    folder_id = os.getenv("DRIVE_FOLDER_ID", "local").strip() or "local"

    rows_csv = ["Arquivo;Codigo;Identificacao;DataHora;Template;Status;Motivo"]
    aprovados_total = 0
    rejeitados_total = 0

    store = Checkpoint(config.resolved_state_dir())
    try:
        for item in resultados:
            nome_arquivo = item.get("file_name") or item.get("file") or "Desconhecido"
            caminho = item.get("file_path")
            fid = item.get("file_id")
            template = item.get("template", "GENERIC_TEMPLATE")
            revision = item.get("md5Checksum") or item.get("modifiedTime") or item.get("source_sha256") or "v1"

            doc = (
                store.discover(fid or nome_arquivo, revision, folder_id, nome_arquivo, "csv")
                if (fid or nome_arquivo)
                else None
            )

            if "error" in item:
                rows_csv.append(
                    _format_csv_line([nome_arquivo, "", "", "", template, "REJEITADO", str(item["error"])])
                )
                rejeitados_total += 1
                if doc:
                    store.update_document(doc["id"], "REVIEW_REQUIRED", str(item["error"]))
                continue

            review_result = reviewer_tool.review_extraction(json.dumps(item, ensure_ascii=False))
            approved_samples = []

            for rev in review_result.get("results", []):
                amostra = rev.get("sample", {})
                codigo = amostra.get("codigo", "")
                ident = amostra.get("identificacao", "")
                data_hora = f"{amostra.get('data', '')} {amostra.get('hora', '')}".strip()
                status = "APROVADO" if rev["approved"] else "REJEITADO"
                motivo = ", ".join(rev["issues"]) if not rev["approved"] else ""
                rows_csv.append(
                    _format_csv_line([nome_arquivo, codigo, ident, data_hora, template, status, motivo])
                )
                if rev["approved"]:
                    aprovados_total += 1
                    approved_samples.append(amostra)
                else:
                    rejeitados_total += 1

            if doc and approved_samples and review_result.get("all_approved"):
                payloads = [
                    {
                        "arquivo": nome_arquivo,
                        "codigo": s.get("codigo", ""),
                        "identificacao": s.get("identificacao", ""),
                        "data_hora": f"{s.get('data', '')} {s.get('hora', '')}".strip(),
                        "template": template,
                        "status": "APROVADO",
                        "motivo": "",
                    }
                    for s in approved_samples
                ]
                store.stage(doc["id"], item, payloads, item.get("source_sha256", ""))

            if caminho and os.path.exists(caminho):
                if review_result.get("all_approved"):
                    feedback_tool.record_success(
                        caminho,
                        template,
                        {},
                        approved_samples,
                        item.get("match_method", "unknown"),
                        float(item.get("confidence", 0.0) or 0.0),
                    )
                else:
                    feedback_tool.record_failure(
                        caminho,
                        template,
                        {},
                        item.get("samples", []),
                        review_result.get("rejection_summary", "Amostras rejeitadas"),
                    )

        atomic_write_text(csv_path, "\n".join(rows_csv) + "\n")

        # Cleanup local transient files & idempotently move Drive items AFTER atomic CSV persistence
        if not config.dry_run:
            for item in resultados:
                caminho = item.get("file_path")
                fid = item.get("file_id")
                nome_arquivo = item.get("file_name") or item.get("file") or ""
                if config.move_drive_files and fid and folder_id != "local":
                    dest_name = "rejeitados" if "error" in item else "processados"
                    try:
                        dest_id = drive_tool.get_or_create_subfolder(folder_id, dest_name)
                        drive_tool.move_to_folder(fid, dest_id, nome_arquivo)
                    except Exception as exc:
                        logger.warning("Falha ao mover %s no Drive: %s", nome_arquivo, exc)
                if caminho and os.path.exists(caminho):
                    try:
                        os.remove(caminho)
                    except OSError as exc:
                        logger.warning("Erro ao remover %s: %s", caminho, exc)
    finally:
        store.close()

    return (
        f"Exportação concluída! Arquivo: {csv_path} | "
        f"Aprovadas: {aprovados_total}, Rejeitadas: {rejeitados_total}."
    )
