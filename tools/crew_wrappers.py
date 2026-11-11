from crewai.tools import tool
import os
import json
import logging
from typing import List, Dict, Any
from .drive_tools import GoogleDriveDownloadTool
from .pattern_identifier import PatternIdentifierTool
from .filler_agent import FillerAgentTool
from .reviewer import Reviewer

logger = logging.getLogger(__name__)

# Instancia as ferramentas legado
drive_legado = GoogleDriveDownloadTool()
identifier_legado = PatternIdentifierTool()
filler_legado = FillerAgentTool()
reviewer_legado = Reviewer()

@tool("Baixar_Arquivos_Drive")
def baixar_arquivos_drive(folder_id: str) -> str:
    """
    Lista e baixa todos os arquivos PDFs e imagens que estão em uma pasta no Google Drive 
    identificada por 'folder_id'. 
    Retorna uma string JSON contendo a lista de dicionários com 'file_id', 'file_name' 
    e 'file_path' local de cada arquivo baixado.
    """
    logger.info(f"Procurando arquivos no folder {folder_id}...")
    arquivos = drive_legado.list_files(folder_id=folder_id)
    
    if not arquivos:
        return json.dumps({"status": "error", "message": "Nenhum arquivo encontrado"})

    arquivos_baixados = []
    for arq in arquivos:
        fid = arq['id']
        fname = arq['name']
        path = drive_legado.download_file(file_id=fid, file_name=fname)
        if path:
            arquivos_baixados.append({"file_id": fid, "file_name": fname, "file_path": path})
            
    return json.dumps({"status": "success", "arquivos": arquivos_baixados})

@tool("Extrair_Dados_Cadeia_Custodia")
def extrair_dados_cadeia_custodia(lista_arquivos_json: str) -> str:
    """
    Recebe um JSON string com a lista de arquivos baixados e faz a identificação do 
    template laboratorial e a leitura de OCR de cada arquivo.
    Retorna os dados extraídos em formato JSON string.
    """
    try:
        dados = json.loads(lista_arquivos_json)
    except json.JSONDecodeError:
        return json.dumps({"status": "error", "message": "Entrada não é um JSON válido"})

    arquivos = dados.get("arquivos", [])
    resultados_totais = []

    for arq in arquivos:
        caminho = arq.get("file_path")
        nome = arq.get("file_name")
        fid = arq.get("file_id")

        if not caminho or not os.path.exists(caminho):
            continue

        # 1. Identificar Template
        id_result_json = identifier_legado._run(file_path=caminho)
        
        try:
            id_data = json.loads(id_result_json)
            if "error" in id_data:
                resultados_totais.append({"file_name": nome, "file_id": fid, "error": id_data["error"]})
                continue
        except Exception:
            resultados_totais.append({"file_name": nome, "file_id": fid, "error": "Erro no parse da identificação"})
            continue

        # 2. Extrair Dados
        ext_result_json = filler_legado._run(file_path=caminho, identification_result=id_result_json)
        
        try:
            ext_data = json.loads(ext_result_json)
            ext_data["file_path"] = caminho
            ext_data["file_id"] = fid
            ext_data["file_name"] = nome
            resultados_totais.append(ext_data)
        except Exception:
            resultados_totais.append({"file_name": nome, "file_id": fid, "error": "Erro no parse da extração"})

    return json.dumps({"status": "success", "resultados": resultados_totais}, ensure_ascii=False)

@tool("Validar_e_Exportar_Dados")
def validar_e_exportar_dados(resultados_extracao_json: str) -> str:
    """
    Analisa os dados (código de amostra, identificação e data), decide quais são válidos, 
    escreve-os no arquivo CSV 'output/resultado_crew.csv' e deleta os arquivos locais processados.
    Retorna um sumário da operação.
    """
    try:
        dados = json.loads(resultados_extracao_json)
        logger.info(f"[DEBUG_VALIDACAO] Tipo recebido do loads: {type(dados)}")
        logger.info(f"[DEBUG_VALIDACAO] Conteúdo: {repr(dados)[:200]}...")
    except Exception as e:
        logger.error(f"Erro no json.loads da validação: {e}")
        return "Erro: Entrada inválida."

    if isinstance(dados, list):
        resultados = dados
    elif isinstance(dados, dict):
        resultados = dados.get("resultados", [])
    else:
        logger.error(f"[DEBUG_VALIDACAO] Tipo inesperado na validação: {type(dados)}")
        resultados = []
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    arquivo_csv = os.path.join(output_dir, "resultado_crew.csv")
    
    aprovados_total = 0
    rejeitados_total = 0

    with open(arquivo_csv, "w", encoding="utf-8") as f:
        f.write("Arquivo;Codigo;Identificacao;DataHora;Template;Status;Motivo\n")

    for item in resultados:
        nome_arquivo = item.get("file", item.get("file_name", "Desconhecido"))
        caminho = item.get("file_path")
        fid = item.get("file_id")
        template = item.get("template", "DESCONHECIDO")
        
        if "error" in item:
            with open(arquivo_csv, "a", encoding="utf-8") as f:
                f.write(f"{nome_arquivo};;;;{template};REJEITADO;{item['error']}\n")
            rejeitados_total += 1
        else:
            # Revisa amostras
            review_result = reviewer_legado.review_extraction(json.dumps(item))
            amostras = item.get("samples", [])
            
            for index, rev in enumerate(review_result.get("results", [])):
                amostra = rev.get("sample", {})
                codigo = amostra.get("codigo", "")
                ident = amostra.get("identificacao", "")
                data_hora = f"{amostra.get('data', '')} {amostra.get('hora', '')}".strip()
                status = "APROVADO" if rev["approved"] else "REJEITADO"
                motivo = ", ".join(rev["issues"]) if not rev["approved"] else ""
                
                with open(arquivo_csv, "a", encoding="utf-8") as f:
                    f.write(f"{nome_arquivo};{codigo};{ident};{data_hora};{template};{status};{motivo}\n")
                
                if rev["approved"]:
                    aprovados_total += 1
                else:
                    rejeitados_total += 1

        # Limpa arquivo local
        if caminho and os.path.exists(caminho):
            try:
                os.remove(caminho)
            except Exception as e:
                logger.warning(f"Erro ao remover {caminho}: {e}")

    return f"Exportação concluída! Aprovadas: {aprovados_total}, Rejeitadas: {rejeitados_total}."
