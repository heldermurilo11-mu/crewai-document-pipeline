import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from crewai.tools import BaseTool
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Caminho do registry
REGISTRY_PATH = Path(__file__).parent.parent / "config" / "template_registry.json"


class PatternIdentifierTool(BaseTool):
    """Identifica o template de uma COC e retorna o mapa de campos para extração.

    Estratégia em duas camadas:
    1. Match rápido por âncoras textuais (regex) — sem custo de API
    2. Fallback: inferência via Gemini quando nenhuma âncora bate — usa API

    O resultado é um FieldMap JSON que será consumido pelo FillerAgentTool.
    """

    name: str = "Pattern Identifier"
    description: str = (
        "Identifica o template de uma Cadeia de Custódia (COC) e retorna "
        "o mapa de campos (field_map) para extração de dados."
    )

    api_key: Optional[str] = None
    confidence_threshold: float = 0.85

    # -----------------------------------------------------------------
    # REGISTRY
    # -----------------------------------------------------------------
    def _load_registry(self) -> Dict[str, Any]:
        """Carrega o template_registry.json."""
        if not REGISTRY_PATH.exists():
            raise FileNotFoundError(f"Registry não encontrado: {REGISTRY_PATH}")

        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Salva o registry atualizado (para templates novos/rascunhos)."""
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        logger.info("💾 Registry atualizado e salvo.")

    # -----------------------------------------------------------------
    # CAMADA 1: MATCH POR ÂNCORAS (rápido, sem API)
    # -----------------------------------------------------------------
    def _extract_text_from_pdf_basic(self, file_path: str) -> str:
        """Tenta extrair texto bruto do PDF para busca de âncoras.

        Usa leitura binária + decodificação UTF-8/Latin-1 como fallback.
        Não é OCR real — é apenas busca textual em PDFs que têm texto embutido.

        Args:
            file_path: Caminho do arquivo PDF.

        Returns:
            Texto bruto extraído (pode ser parcial ou vazio para PDFs escaneados).
        """
        try:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()

            # Tenta decodificar como UTF-8
            try:
                text = raw_bytes.decode("utf-8", errors="ignore")
            except Exception:
                text = raw_bytes.decode("latin-1", errors="ignore")

            # Filtra apenas caracteres imprimíveis e espaços
            text = "".join(c if c.isprintable() or c in "\n\r\t" else " " for c in text)
            return text

        except Exception as e:
            logger.warning(f"⚠️ Não foi possível ler texto do PDF: {e}")
            return ""

    def _match_by_anchors(self, text: str, templates: List[Dict]) -> Tuple[Optional[Dict], float]:
        """Busca âncoras dos templates no texto extraído do PDF.

        Percorre os templates na ordem de prioridade e verifica se as
        âncoras existem no texto do documento.

        Args:
            text: Texto bruto extraído do PDF.
            templates: Lista de templates do registry, já ordenados por prioridade.

        Returns:
            Tupla (template_match, confidence). Se não encontrar, retorna (None, 0.0).
        """
        if not text or len(text.strip()) < 50:
            # Texto muito curto — provavelmente PDF escaneado sem texto embutido
            return None, 0.0

        text_upper = text.upper()

        for tmpl in sorted(templates, key=lambda t: t.get("priority", 99)):
            anchors = tmpl.get("anchors", [])
            total_weight = sum(a.get("weight", 1.0) for a in anchors)
            matched_weight = 0.0

            for anchor in anchors:
                anchor_text = anchor["text"].upper()
                weight = anchor.get("weight", 1.0)

                # Busca flexível: ignora espaços extras e variações
                pattern = re.escape(anchor_text).replace(r"\ ", r"\s+")
                if re.search(pattern, text_upper):
                    matched_weight += weight

            if total_weight > 0:
                confidence = matched_weight / total_weight
                if confidence >= self.confidence_threshold:
                    logger.info(
                        f"🔍 Match por âncoras: {tmpl['name']} "
                        f"(confiança: {confidence:.2f})"
                    )
                    return tmpl, confidence

        return None, 0.0

    # -----------------------------------------------------------------
    # CAMADA 2: INFERÊNCIA VIA GEMINI (fallback para templates novos)
    # -----------------------------------------------------------------
    def _infer_via_gemini(self, file_path: str) -> Optional[Dict]:
        """Usa Gemini para inferir o layout de um template desconhecido.

        Envia o PDF ao Gemini com um prompt de análise de layout e pede
        que ele proponha um field_map para os campos desejados.

        Args:
            file_path: Caminho do arquivo PDF.

        Returns:
            Dicionário com template inferido, ou None se falhar.
        """
        chave = self.api_key if self.api_key else os.getenv("GEMINI_API_KEY")
        if not chave:
            logger.error("❌ GEMINI_API_KEY não configurada para inferência.")
            return None

        client = genai.Client(api_key=chave)

        inference_prompt = """Você é um especialista em análise de layout de documentos laboratoriais.

Analise o documento PDF fornecido. Ele é uma Cadeia de Custódia (COC) de um laboratório ou consultoria ambiental.

Sua tarefa é MAPEAR ONDE estão os seguintes campos neste documento:
1. **codigo**: Número de 7 dígitos (geralmente manuscrito a caneta) que identifica a amostra.
2. **identificacao**: Texto ou código que descreve o ponto/nome da amostra.
3. **data**: Data da coleta da amostra (formato DD/MM/AAAA).
4. **hora**: Hora da coleta (formato HH:MM).

Para CADA campo, informe:
- "location": Descrição textual de ONDE o campo está no documento (ex: "coluna 'Identificação' na tabela principal")
- "anchor": Qual texto impresso no documento serve como referência para localizar esse campo
- "type": "printed" (impresso), "handwritten" (manuscrito) ou "printed_or_handwritten"
- "format": Formato esperado do valor (ex: "7 dígitos", "DD/MM/AAAA")

Também identifique:
- "detected_name": Nome/marca do laboratório ou empresa que aparece no documento
- "detected_anchors": Lista de textos impressos únicos que poderiam identificar este tipo de documento
- "special_rules": Lista de regras ou observações importantes para extração correta

IMPORTANTE:
- Foque no layout REAL do documento
- Se algum campo tiver significado invertido (ex: "Código Amostra" na verdade contém a identificação), documente isso em special_rules
- Se data e hora compartilham o mesmo campo, documente isso

Retorne APENAS um JSON válido com esta estrutura:
{
  "detected_name": "NOME DO LABORATÓRIO OU MODELO",
  "confidence": 0.7,
  "detected_anchors": [
    {"text": "TEXTO ÂNCORA IDENTIFICADO", "location": "cabeçalho", "weight": 1.0}
  ],
  "field_map": {
    "codigo": {"location": "...", "anchor": "...", "type": "...", "format": "7 dígitos"},
    "identificacao": {"location": "...", "anchor": "...", "type": "...", "format": "..."},
    "data": {"location": "...", "anchor": "...", "type": "...", "format": "DD/MM/AAAA"},
    "hora": {"location": "...", "anchor": "...", "type": "...", "format": "HH:MM"}
  },
  "special_rules": ["..."]
}
"""

        try:
            with open(file_path, "rb") as f:
                file_content = f.read()

            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(data=file_content, mime_type="application/pdf"),
                    inference_prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )

            if not response.text:
                logger.warning("⚠️ Gemini retornou resposta vazia na inferência de layout.")
                return None

            inferred = json.loads(response.text)
            logger.info(
                f"🧠 Gemini inferiu template: {inferred.get('detected_name', 'DESCONHECIDO')} "
                f"(confiança: {inferred.get('confidence', 0):.2f})"
            )
            return inferred

        except Exception as e:
            logger.error(f"❌ Erro na inferência via Gemini: {e}")
            return None

    def _register_draft_template(self, registry: Dict, inferred: Dict) -> Dict:
        """Registra um template inferido como rascunho no registry.

        Args:
            registry: Registry atual.
            inferred: Dados inferidos pelo Gemini.

        Returns:
            O template criado em formato do registry.
        """
        name = (
            inferred.get("detected_name", "DESCONHECIDO")
            .upper()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("Í", "I")
        )

        # Verifica se já existe um rascunho com esse nome
        existing_names = {t["name"] for t in registry.get("templates", [])}
        if name in existing_names:
            logger.info(f"📋 Template rascunho '{name}' já existe no registry. Atualizando.")
            for tmpl in registry["templates"]:
                if tmpl["name"] == name:
                    # Atualiza o field_map com a nova inferência
                    tmpl["field_map"] = inferred.get("field_map", tmpl["field_map"])
                    if inferred.get("special_rules"):
                        tmpl["special_rules"] = inferred["special_rules"]
                    return tmpl

        # Calcula próxima prioridade
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
        logger.info(f"📝 Template rascunho '{name}' registrado com prioridade {new_template['priority']}.")

        return new_template

    # -----------------------------------------------------------------
    # EXECUÇÃO PRINCIPAL
    # -----------------------------------------------------------------
    def _run(self, file_path: str) -> str:
        """Identifica o template de uma COC e retorna o field_map como JSON.

        Estratégia:
        1. Tenta match por âncoras (rápido, sem API)
        2. Se falhar, usa Gemini para inferir o layout (com custo de API)
        3. Templates novos são salvos como rascunho no registry

        Args:
            file_path: Caminho absoluto do arquivo PDF.

        Returns:
            JSON string com o resultado da identificação (template + field_map).
        """
        if not os.path.exists(file_path):
            return json.dumps({"error": f"Arquivo {file_path} não encontrado."})

        nome_arquivo = os.path.basename(file_path)

        try:
            registry = self._load_registry()
        except FileNotFoundError as e:
            return json.dumps({"error": str(e)})

        templates = registry.get("templates", [])

        # --- Camada 1: Match por Âncoras ---
        text = self._extract_text_from_pdf_basic(file_path)
        matched_template, confidence = self._match_by_anchors(text, templates)

        if matched_template:
            result = {
                "template_name": matched_template["name"],
                "confidence": round(confidence, 2),
                "field_map": matched_template.get("field_map", {}),
                "special_rules": matched_template.get("special_rules", []),
                "is_new_template": False,
                "status": matched_template.get("status", "unknown"),
                "match_method": "anchor",
            }
            logger.info(f"✅ {nome_arquivo}: Template identificado por âncoras → {matched_template['name']}")
            return json.dumps(result, ensure_ascii=False)

        # --- Camada 2: Inferência via Gemini ---
        logger.info(f"🔍 {nome_arquivo}: Nenhuma âncora bateu. Iniciando inferência via Gemini...")

        inferred = self._infer_via_gemini(file_path)

        if inferred:
            # Registra como rascunho
            new_template = self._register_draft_template(registry, inferred)

            # Salva registry atualizado
            try:
                self._save_registry(registry)
            except Exception as e:
                logger.warning(f"⚠️ Não foi possível salvar registry: {e}")

            result = {
                "template_name": new_template["name"],
                "confidence": round(inferred.get("confidence", 0.5), 2),
                "field_map": inferred.get("field_map", {}),
                "special_rules": inferred.get("special_rules", []),
                "is_new_template": True,
                "status": "draft",
                "match_method": "gemini_inference",
            }
            logger.info(f"🧠 {nome_arquivo}: Template inferido pelo Gemini → {new_template['name']}")
            return json.dumps(result, ensure_ascii=False)

        # --- Fallback: Genérico ---
        generic = registry.get("generic_fallback", {})
        result = {
            "template_name": generic.get("template_name", "GENERICO"),
            "confidence": 0.0,
            "field_map": {},
            "special_rules": generic.get("rules", []),
            "is_new_template": False,
            "status": "generic_fallback",
            "match_method": "none",
        }
        logger.warning(f"⚠️ {nome_arquivo}: Não foi possível identificar o template. Usando GENÉRICO.")
        return json.dumps(result, ensure_ascii=False)
