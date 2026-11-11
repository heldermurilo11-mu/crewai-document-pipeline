"""Parameterized high-DPI document extractor built on Gemini Vision and runtime guards."""

from dataclasses import replace
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from runtime.config import RuntimeConfig
from runtime.deadline import Deadline
from runtime.extraction import extract_document
from ._crew_compat import BaseTool
from .utils import get_configured_prefixes, normalize_code, normalize_time

load_dotenv()
logger = logging.getLogger(__name__)


class FillerAgentTool(BaseTool):
    """Extrai dados estruturados usando renderização 300 DPI, orientação e schema estrito."""

    name: str = "Filler Agent"
    description: str = (
        "Extrai dados concretos (codigo, identificacao, data, hora) de uma COC "
        "usando renderização de alta resolução, detecção de orientação e schema estrito."
    )

    api_key: Optional[str] = None
    max_retries: int = 2
    expected_code_prefixes: Optional[List[str]] = None
    runtime_config: Optional[RuntimeConfig] = None

    def _build_extraction_prompt(
        self,
        template_name: str,
        field_map: Dict[str, Any],
        special_rules: List[str],
    ) -> str:
        current_year = datetime.now().year
        short_year = str(current_year)[-2:]
        prefixes = get_configured_prefixes(self.expected_code_prefixes)
        prefix_hint = (
            f"\n   - Prefixos esperados configurados: {', '.join(repr(p) for p in prefixes)}."
            if prefixes
            else ""
        )

        prompt = (
            "Você é um especialista em extração de dados de formulários laboratoriais.\n"
            f"O documento corresponde ao template **{template_name}**.\n\n"
            "### INSTRUÇÕES DE LOCALIZAÇÃO DOS CAMPOS\n"
        )
        if field_map:
            for field_name, field_info in field_map.items():
                if not isinstance(field_info, dict):
                    continue
                location = field_info.get("location", "não especificado")
                fmt = field_info.get("format", "")
                notes = field_info.get("notes", "")
                field_type = field_info.get("type", "")
                prompt += f"**{field_name}**:\n- Localização: {location}\n"
                if fmt:
                    prompt += f"- Formato esperado: {fmt}\n"
                if field_type:
                    prompt += f"- Tipo: {field_type}\n"
                if notes:
                    prompt += f"- Atenção: {notes}\n"
                for src_name, src_info in (field_info.get("sources") or {}).items():
                    prompt += (
                        f'  - Fonte "{src_name}": campo "{src_info.get("anchor", "")}" '
                        f'(formato: {src_info.get("format", "")})\n'
                    )
                aliases = field_info.get("anchor_aliases") or []
                if aliases:
                    prompt += f"- Rótulos equivalentes: {', '.join(aliases)}\n"
                prompt += "\n"
        else:
            prompt += (
                "- Analise todas as tabelas de amostras no documento e extraia codigo, "
                "identificacao, data e hora de cada linha.\n\n"
            )

        if special_rules:
            prompt += "### REGRAS ESPECIAIS DO TEMPLATE\n"
            for idx, rule in enumerate(special_rules, 1):
                prompt += f"{idx}. {rule}\n"
            prompt += "\n"

        prompt += f"""### REGRAS GERAIS DE EXTRAÇÃO
1. Ignore linhas verticais/bordas de células manuscritas (não confunda bordas com '1' ou '7').
2. Quando presente, `codigo` numérico de amostra tem tipicamente 7 dígitos.{prefix_hint}
3. Preserve datas em **DD/MM/AAAA** (converta ano curto ex: "{short_year}" para {current_year} quando coerente com o documento).
4. Se a coluna Data de uma linha estiver vazia, utilize a data global do cabeçalho ou rodapé do próprio documento caso explícita.
5. Analise TODAS as páginas em ordem e retorne UM único objeto JSON conforme o schema fornecido.
"""
        return prompt

    def extract(
        self,
        file_path: str | Path,
        *,
        identification_result: Optional[Dict[str, Any]] = None,
        deadline: Optional[Deadline] = None,
        work_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run high-DPI, orientation-aware, schema-enforced extraction on *file_path*."""
        ident = identification_result or {}
        template_name = ident.get("template_name", "GENERIC_TEMPLATE")
        field_map = ident.get("field_map", {})
        special_rules = ident.get("special_rules", [])
        match_method = ident.get("match_method", "unknown")
        confidence = float(ident.get("confidence", 0.0) or 0.0)

        config = self.runtime_config or RuntimeConfig.from_env()
        config = replace(config, extraction_max_attempts=max(1, int(self.max_retries)))
        config.validate()
        active_deadline = deadline or Deadline(config.file_budget_seconds)
        prefixes = get_configured_prefixes(self.expected_code_prefixes)

        return extract_document(
            file_path=file_path,
            api_key=self.api_key or os.getenv("GEMINI_API_KEY", ""),
            prompt=self._build_extraction_prompt(template_name, field_map, special_rules),
            config=config,
            deadline=active_deadline,
            normalize_code_fn=lambda code: normalize_code(code, known_prefixes=prefixes),
            normalize_time_fn=normalize_time,
            work_dir=work_dir,
            template_name=template_name,
            match_method=match_method,
            confidence=confidence,
        )

    def _run(self, file_path: str, identification_result: str = "{}") -> str:
        try:
            id_result = json.loads(identification_result) if identification_result else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"JSON de identificação inválido: {exc}", "samples": []})

        try:
            result = self.extract(file_path, identification_result=id_result)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.error("Falha na extração de %s: %s", file_path, exc)
            return json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "template": id_result.get("template_name", "GENERIC_TEMPLATE"),
                    "samples": [],
                },
                ensure_ascii=False,
            )
