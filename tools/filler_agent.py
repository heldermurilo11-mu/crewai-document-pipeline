import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from crewai.tools import BaseTool
from google import genai
from google.genai import types
from dotenv import load_dotenv

from .utils import get_configured_prefixes, normalize_time, normalize_code

load_dotenv()
logger = logging.getLogger(__name__)


class FillerAgentTool(BaseTool):
    """Extrai dados concretos de uma COC usando um prompt genérico parametrizado.

    Recebe o field_map do PatternIdentifierTool e usa-o para construir
    um prompt Gemini que sabe ONDE procurar cada campo no documento.
    """

    name: str = "Filler Agent"
    description: str = (
        "Extrai dados concretos (codigo, identificacao, data, hora) de uma COC "
        "usando um mapa de campos (field_map) que indica onde cada informação está."
    )

    api_key: Optional[str] = None
    max_retries: int = 3
    expected_code_prefixes: Optional[List[str]] = None

    # -----------------------------------------------------------------
    # CONSTRUÇÃO DO PROMPT PARAMETRIZADO
    # -----------------------------------------------------------------
    def _build_extraction_prompt(
        self,
        template_name: str,
        field_map: Dict[str, Any],
        special_rules: List[str],
    ) -> str:
        """Monta o prompt de extração parametrizado pelo field_map.

        Em vez de ter regras fixas por template, o prompt é genérico e recebe
        as instruções de localização de cada campo como parâmetros.

        Args:
            template_name: Nome do template detectado (ex: "LAB_ALPHA").
            field_map: Mapa de campos com localização de cada dado.
            special_rules: Regras especiais para este template.

        Returns:
            Prompt completo para enviar ao Gemini.
        """
        prompt = f"""Você é um especialista em extração de dados de formulários laboratoriais.

O documento é uma Cadeia de Custódia (COC) do template **{template_name}**.

### INSTRUÇÕES DE LOCALIZAÇÃO DOS CAMPOS

Para cada amostra encontrada no documento, extraia os seguintes campos:

"""
        # Instruções por campo
        for field_name, field_info in field_map.items():
            if isinstance(field_info, dict):
                location = field_info.get("location", "não especificado")
                fmt = field_info.get("format", "")
                notes = field_info.get("notes", "")
                field_type = field_info.get("type", "")

                prompt += f'**{field_name}**:\n'
                prompt += f'  - Localização: {location}\n'
                if fmt:
                    prompt += f'  - Formato esperado: {fmt}\n'
                if field_type:
                    type_desc = {
                        "handwritten": "manuscrito a caneta",
                        "printed": "impresso/digitado",
                        "printed_or_handwritten": "pode ser impresso ou manuscrito",
                        "computed": "valor combinado de outros campos",
                    }.get(field_type, field_type)
                    prompt += f'  - Tipo: {type_desc}\n'
                if notes:
                    prompt += f'  - ATENÇÃO: {notes}\n'

                # Se o campo tem sources (para campos compostos, ex.: identificacao do LAB_BETA)
                sources = field_info.get("sources", {})
                if sources:
                    prompt += '  - Fontes para composição:\n'
                    for src_name, src_info in sources.items():
                        src_anchor = src_info.get("anchor", "")
                        src_fmt = src_info.get("format", "")
                        prompt += f'    - "{src_name}": campo "{src_anchor}" (formato: {src_fmt})\n'

                # Anchor aliases
                aliases = field_info.get("anchor_aliases", [])
                if aliases:
                    aliases_str = ", ".join(f'"{a}"' for a in aliases)
                    prompt += f'  - Rótulos equivalentes aceitos: [{aliases_str}]\n'

                prompt += "\n"

        # Regras especiais
        if special_rules:
            prompt += "### REGRAS ESPECIAIS PARA ESTE TEMPLATE\n\n"
            for i, rule in enumerate(special_rules, 1):
                prompt += f'{i}. {rule}\n'
            prompt += "\n"

        prefixes = get_configured_prefixes(self.expected_code_prefixes)
        prefix_hint = (
            f"\n   - Prefixos configurados esperados para códigos válidos: {', '.join(repr(p) for p in prefixes)}."
            if prefixes
            else ""
        )
        current_year = datetime.now().year
        short_year = str(current_year)[-2:]

        # Regras gerais e formato de saída
        prompt += f"""### REGRAS GERAIS DE EXTRAÇÃO (CRÍTICO)
1. **Deduplicação de Bordas:** Ao ler números manuscritos dentro de tabelas, ignore COMPLETAMENTE as linhas verticais (bordas) das células. Elas são frequentemente confundidas com o dígito '1' ou '7'.
2. **Regra de 7 Dígitos:** O "codigo" de amostra deve ter **EXATAMENTE 7 dígitos numéricos**.
3. **Prefixos e Ruídos:**
   - Se ler 8 dígitos começando com '1' ou '0', verifique se o primeiro dígito não é um ruído (borda ou marcação externa) e extraia apenas os 7 dígitos válidos.{prefix_hint}
4. **Data e Ano (IMPORTANTE):**
   - Extraia a data SEMPRE no formato **DD/MM/AAAA**.
   - Se o documento mostrar apenas o ano com 2 dígitos (ex: "{short_year}"), converta automaticamente para o formato completo: **{current_year}**.
5. **Dígitos Pequenos:** Atenção a dígitos pequenos ou "esprimidos" dentro das células. Não pule nenhum número.

### FORMATO DE SAÍDA
Retorne APENAS um JSON válido seguindo estritamente esta estrutura:
[
  {{
    "template": "{template_name}",
    "codigo": "XXXXXXX (exatamente 7 dígitos)",
    "identificacao": "texto_extraido",
    "data": "DD/MM/AAAA (sempre 4 dígitos no ano)",
    "hora": "HH:MM (formato 24h)"
  }}
]

"""
        return prompt

    # -----------------------------------------------------------------
    # EXECUÇÃO PRINCIPAL
    # -----------------------------------------------------------------
    def _run(self, file_path: str, identification_result: str) -> str:
        """Extrai dados concretos de uma COC usando o field_map fornecido.

        Args:
            file_path: Caminho absoluto do arquivo PDF.
            identification_result: JSON string com o resultado do PatternIdentifierTool
                                   (contém template_name, field_map, special_rules).

        Returns:
            JSON string com os dados extraídos ou mensagem de erro.
        """
        if not os.path.exists(file_path):
            return json.dumps({"error": f"Arquivo {file_path} não encontrado."})

        nome_arquivo = os.path.basename(file_path)

        # Parse do resultado de identificação
        try:
            id_result = json.loads(identification_result)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"JSON de identificação inválido: {e}"})

        template_name = id_result.get("template_name", "GENERICO")
        field_map = id_result.get("field_map", {})
        special_rules = id_result.get("special_rules", [])

        # Se não tem field_map, não consegue extrair
        if not field_map:
            logger.warning(f"⚠️ {nome_arquivo}: field_map vazio. Extração impossível.")
            return json.dumps({
                "error": "field_map vazio — extração impossível",
                "template": template_name,
                "samples": [],
            })

        # Montar prompt parametrizado
        prompt = self._build_extraction_prompt(template_name, field_map, special_rules)

        # Configuração do Gemini
        chave = self.api_key if self.api_key else os.getenv("GEMINI_API_KEY")
        if not chave:
            return json.dumps({"error": "GEMINI_API_KEY não configurada."})

        client = genai.Client(api_key=chave)
        prefixes = get_configured_prefixes(self.expected_code_prefixes)

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                time.sleep(2)  # Rate limit

                with open(file_path, "rb") as f:
                    file_content = f.read()

                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[
                        types.Part.from_bytes(data=file_content, mime_type="application/pdf"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )

                if not response.text:
                    logger.warning(f"⚠️ {nome_arquivo}: Resposta vazia (tentativa {attempt + 1}).")
                    continue

                dados_json = json.loads(response.text)

                # Normaliza para lista
                if isinstance(dados_json, dict):
                    dados_json = [dados_json]

                # Validação e limpeza
                samples = []
                for item in dados_json:
                    sample = {
                        "template": str(item.get("template", template_name)).strip(),
                        "codigo": normalize_code(
                            item.get("codigo", ""),
                            known_prefixes=prefixes,
                        ),
                        "identificacao": (
                            str(item.get("identificacao", ""))
                            .replace(";", ",")
                            .replace("\\n", " ")
                            .strip()
                        ),
                        "data": str(item.get("data", "")).strip(),
                        "hora": normalize_time(str(item.get("hora", ""))),
                    }
                    # Só aceita se tiver código ou identificação
                    if sample["codigo"] or sample["identificacao"]:
                        samples.append(sample)

                result = {
                    "template": template_name,
                    "file": nome_arquivo,
                    "samples": samples,
                    "total_extracted": len(samples),
                    "match_method": id_result.get("match_method", "unknown"),
                    "confidence": id_result.get("confidence", 0),
                }

                if samples:
                    logger.info(
                        f"✅ {nome_arquivo}: {len(samples)} amostras extraídas "
                        f"(template: {template_name})"
                    )
                else:
                    logger.warning(f"⚠️ {nome_arquivo}: Nenhuma amostra válida encontrada.")

                return json.dumps(result, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Erro tentativa {attempt + 1} em {nome_arquivo}: {e}")
                if attempt == self.max_retries - 1:
                    return json.dumps({
                        "error": f"Falha após {self.max_retries} tentativas: {str(e)}",
                        "template": template_name,
                        "samples": [],
                    })
                time.sleep(5 * (attempt + 1))

        return json.dumps({"error": "Erro desconhecido.", "template": template_name, "samples": []})
