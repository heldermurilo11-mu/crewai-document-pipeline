"""Two-layer template identification: PyMuPDF anchor matching + Gemini layout inference."""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
import pymupdf

from runtime.deadline import Deadline
from runtime.files import atomic_json
from ._crew_compat import BaseTool

load_dotenv()
logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "template_registry.json"


class PatternIdentifierTool(BaseTool):
    """Identifica o template de uma COC e retorna o mapa de campos para extração."""

    name: str = "Pattern Identifier"
    description: str = (
        "Identifica o template de uma Cadeia de Custódia (COC) e retorna "
        "o mapa de campos (field_map) para extração de dados."
    )

    api_key: Optional[str] = None
    confidence_threshold: float = 0.85
    registry_path: Path = REGISTRY_PATH

    def _load_registry(self) -> Dict[str, Any]:
        target = Path(self.registry_path)
        if not target.exists():
            raise FileNotFoundError(f"Registry não encontrado: {target}")
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        atomic_json(self.registry_path, registry)
        logger.info("Registry atualizado atomicamente em %s", self.registry_path)

    def _extract_text_from_pdf_basic(self, file_path: str) -> str:
        """Extract embedded text via PyMuPDF cleanly before touching Gemini Vision."""
        path = Path(file_path)
        if path.suffix.lower() != ".pdf":
            return ""
        try:
            with pymupdf.open(str(path)) as doc:
                return "\n".join(page.get_text("text") or "" for page in doc)
        except Exception as exc:
            logger.warning("Não foi possível ler texto via PyMuPDF em %s: %s", path.name, exc)
            return ""

    def _match_by_anchors(
        self, text: str, templates: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        if not text or len(text.strip()) < 20:
            return None, 0.0

        text_upper = text.upper()
        for tmpl in sorted(templates, key=lambda t: t.get("priority", 99)):
            anchors = tmpl.get("anchors", [])
            total_weight = sum(a.get("weight", 1.0) for a in anchors)
            matched_weight = 0.0

            for anchor in anchors:
                anchor_text = anchor["text"].upper()
                weight = anchor.get("weight", 1.0)
                pattern = re.escape(anchor_text).replace(r"\ ", r"\s+")
                if re.search(pattern, text_upper):
                    matched_weight += weight

            if total_weight > 0:
                confidence = matched_weight / total_weight
                if confidence >= self.confidence_threshold:
                    logger.info("Match por âncoras: %s (confiança=%.2f)", tmpl["name"], confidence)
                    return tmpl, confidence

        return None, 0.0

    def _infer_via_gemini(
        self,
        file_path: str,
        deadline: Optional[Deadline] = None,
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        chave = self.api_key or os.getenv("GEMINI_API_KEY")
        if not chave:
            logger.error("GEMINI_API_KEY não configurada para inferência de template.")
            return None

        inference_prompt = """Você é um especialista em análise de layout de documentos laboratoriais.

Analise o documento fornecido (Cadeia de Custódia / formulário tabular).
Mapeie ONDE estão os seguintes campos:
1. codigo: Código numérico identificador da amostra (7 dígitos quando aplicável).
2. identificacao: Texto ou ponto de identificação da amostra.
3. data: Data de coleta (DD/MM/AAAA).
4. hora: Hora de coleta (HH:MM).

Retorne APENAS um JSON válido com:
{
  "detected_name": "NOME DO MODELO OU TEMPLATE",
  "confidence": 0.7,
  "detected_anchors": [
    {"text": "TEXTO ÂNCORA", "location": "cabeçalho", "weight": 1.0}
  ],
  "field_map": {
    "codigo": {"location": "...", "anchor": "...", "type": "handwritten", "format": "7 dígitos"},
    "identificacao": {"location": "...", "anchor": "...", "type": "printed", "format": "texto"},
    "data": {"location": "...", "anchor": "...", "type": "printed_or_handwritten", "format": "DD/MM/AAAA"},
    "hora": {"location": "...", "anchor": "...", "type": "printed_or_handwritten", "format": "HH:MM"}
  },
  "special_rules": ["..."]
}
"""
        client = genai.Client(api_key=chave)
        try:
            path = Path(file_path)
            mime_type = "application/pdf" if path.suffix.lower() == ".pdf" else "image/png"
            config_kwargs: Dict[str, Any] = {
                "response_mime_type": "application/json",
                "temperature": 0.2,
            }
            if deadline is not None:
                config_kwargs["http_options"] = types.HttpOptions(
                    timeout=deadline.timeout_ms(120_000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                )
            model = model_name or os.getenv("GEMINI_MODEL_FAST", "gemini-2.5-flash")
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type),
                    inference_prompt,
                ],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            if not getattr(response, "text", None):
                return None
            return json.loads(response.text)
        except Exception as exc:
            logger.error("Erro na inferência via Gemini: %s", exc)
            return None
        finally:
            if hasattr(client, "close") and callable(client.close):
                try:
                    client.close()
                except Exception:
                    pass

    def _register_draft_template(self, registry: Dict[str, Any], inferred: Dict[str, Any]) -> Dict[str, Any]:
        name = (
            str(inferred.get("detected_name", "DESCONHECIDO"))
            .upper()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("Í", "I")
        )
        for tmpl in registry.get("templates", []):
            if tmpl["name"] == name:
                tmpl["field_map"] = inferred.get("field_map", tmpl["field_map"])
                if inferred.get("special_rules"):
                    tmpl["special_rules"] = inferred["special_rules"]
                return tmpl

        max_priority = max((t.get("priority", 0) for t in registry.get("templates", [])), default=0)
        new_template = {
            "name": name,
            "priority": max_priority + 1,
            "status": "draft",
            "anchors": inferred.get("detected_anchors", []),
            "field_map": inferred.get("field_map", {}),
            "special_rules": inferred.get("special_rules", []),
            "confidence_history": [inferred.get("confidence", 0.5)],
            "total_validations": 0,
            "total_rejections": 0,
        }
        registry.setdefault("templates", []).append(new_template)
        return new_template

    def identify(
        self,
        file_path: str,
        *,
        deadline: Optional[Deadline] = None,
        allow_inference: bool = True,
    ) -> Dict[str, Any]:
        """Return structured identification dictionary for *file_path*."""
        if not os.path.exists(file_path):
            return {"error": f"Arquivo {file_path} não encontrado."}

        registry = self._load_registry()
        templates = registry.get("templates", [])
        text = self._extract_text_from_pdf_basic(file_path)
        matched_template, confidence = self._match_by_anchors(text, templates)

        if matched_template:
            return {
                "template_name": matched_template["name"],
                "confidence": round(confidence, 2),
                "field_map": matched_template.get("field_map", {}),
                "special_rules": matched_template.get("special_rules", []),
                "is_new_template": False,
                "status": matched_template.get("status", "unknown"),
                "match_method": "anchor",
            }

        if allow_inference and (self.api_key or os.getenv("GEMINI_API_KEY")):
            inferred = self._infer_via_gemini(file_path, deadline=deadline)
            if inferred:
                new_template = self._register_draft_template(registry, inferred)
                try:
                    self._save_registry(registry)
                except Exception as exc:
                    logger.warning("Não foi possível salvar registry: %s", exc)
                return {
                    "template_name": new_template["name"],
                    "confidence": round(inferred.get("confidence", 0.5), 2),
                    "field_map": inferred.get("field_map", {}),
                    "special_rules": inferred.get("special_rules", []),
                    "is_new_template": True,
                    "status": "draft",
                    "match_method": "gemini_inference",
                }

        generic = registry.get("generic_fallback", {})
        return {
            "template_name": generic.get("template_name", "GENERIC_TEMPLATE"),
            "confidence": 0.0,
            "field_map": {},
            "special_rules": generic.get("rules", []),
            "is_new_template": False,
            "status": "generic_fallback",
            "match_method": "none",
        }

    def _run(self, file_path: str) -> str:
        return json.dumps(self.identify(file_path), ensure_ascii=False)
