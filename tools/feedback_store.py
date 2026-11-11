import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Diretório para persistência de feedback
FEEDBACK_DIR = Path(__file__).parent.parent / "output" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback_log.json"


class FeedbackStore:
    """Persiste feedback de extrações para evolução do sistema.

    Armazena um registro para cada extração realizada, incluindo:
    - Template usado, field_map, dados extraídos, status da revisão
    - Correções feitas pelo revisor (se houver)

    Esses dados alimentam a promoção de templates (rascunho → validado)
    e futuramente servirão como dataset para treinamento de ML.
    """

    def __init__(self, feedback_dir: Optional[Path] = None):
        self.feedback_dir = feedback_dir or FEEDBACK_DIR
        self.feedback_file = self.feedback_dir / "feedback_log.json"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        # Inicializa o arquivo de feedback se não existir
        if not self.feedback_file.exists():
            self._save_data({"entries": [], "stats": {}})

    def _load_data(self) -> Dict[str, Any]:
        """Carrega o arquivo de feedback."""
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"entries": [], "stats": {}}

    def _save_data(self, data: Dict[str, Any]) -> None:
        """Salva o arquivo de feedback."""
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _file_hash(self, file_path: str) -> str:
        """Gera hash SHA-256 do arquivo para identificação única."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return "unknown"

    # -----------------------------------------------------------------
    # REGISTRAR RESULTADOS
    # -----------------------------------------------------------------
    def record_success(
        self,
        file_path: str,
        template_name: str,
        field_map_used: Dict,
        extraction_result: List[Dict],
        match_method: str,
        confidence: float,
        reviewer: str = "auto",
    ) -> None:
        """Registra uma extração bem-sucedida (aprovada pela revisão).

        Args:
            file_path: Caminho do PDF processado.
            template_name: Template usado na extração.
            field_map_used: Mapa de campos usado.
            extraction_result: Lista de amostras extraídas.
            match_method: Método de match usado (anchor/gemini_inference).
            confidence: Confiança da identificação.
            reviewer: Quem revisou (auto/human).
        """
        data = self._load_data()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "file": os.path.basename(file_path),
            "file_hash": self._file_hash(file_path),
            "template": template_name,
            "match_method": match_method,
            "confidence": confidence,
            "review_status": "approved",
            "reviewer": reviewer,
            "samples_extracted": len(extraction_result),
            "corrections": None,
        }

        data["entries"].append(entry)
        self._update_stats(data, template_name, success=True)
        self._save_data(data)

        logger.info(
            f"📊 Feedback registrado: {template_name} | "
            f"aprovado | {len(extraction_result)} amostras"
        )

    def record_failure(
        self,
        file_path: str,
        template_name: str,
        field_map_used: Dict,
        extraction_result: List[Dict],
        error_description: str,
        corrections: Optional[Dict] = None,
        reviewer: str = "auto",
    ) -> None:
        """Registra uma extração rejeitada pela revisão.

        Args:
            file_path: Caminho do PDF processado.
            template_name: Template usado.
            field_map_used: Mapa de campos usado.
            extraction_result: Dados extraídos (incorretos).
            error_description: Descrição do erro encontrado.
            corrections: Correções feitas pelo revisor (se houver).
            reviewer: Quem revisou.
        """
        data = self._load_data()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "file": os.path.basename(file_path),
            "file_hash": self._file_hash(file_path),
            "template": template_name,
            "match_method": "unknown",
            "confidence": 0,
            "review_status": "rejected",
            "reviewer": reviewer,
            "samples_extracted": len(extraction_result),
            "error": error_description,
            "corrections": corrections,
        }

        data["entries"].append(entry)
        self._update_stats(data, template_name, success=False)
        self._save_data(data)

        logger.warning(
            f"📊 Feedback registrado: {template_name} | "
            f"rejeitado | erro: {error_description[:60]}"
        )

    # -----------------------------------------------------------------
    # ESTATÍSTICAS
    # -----------------------------------------------------------------
    def _update_stats(self, data: Dict, template_name: str, success: bool) -> None:
        """Atualiza as estatísticas por template."""
        stats = data.setdefault("stats", {})
        tmpl_stats = stats.setdefault(template_name, {
            "total_extractions": 0,
            "approved": 0,
            "rejected": 0,
            "success_rate": 0.0,
            "consecutive_successes": 0,
            "consecutive_failures": 0,
        })

        tmpl_stats["total_extractions"] += 1

        if success:
            tmpl_stats["approved"] += 1
            tmpl_stats["consecutive_successes"] += 1
            tmpl_stats["consecutive_failures"] = 0
        else:
            tmpl_stats["rejected"] += 1
            tmpl_stats["consecutive_failures"] += 1
            tmpl_stats["consecutive_successes"] = 0

        total = tmpl_stats["total_extractions"]
        tmpl_stats["success_rate"] = round(tmpl_stats["approved"] / total, 2) if total > 0 else 0.0

    def get_template_stats(self, template_name: str) -> Dict[str, Any]:
        """Retorna as estatísticas de um template específico.

        Args:
            template_name: Nome do template.

        Returns:
            Dicionário com estatísticas do template.
        """
        data = self._load_data()
        return data.get("stats", {}).get(template_name, {
            "total_extractions": 0,
            "approved": 0,
            "rejected": 0,
            "success_rate": 0.0,
        })

    def get_all_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de todos os templates."""
        data = self._load_data()
        return data.get("stats", {})

    def should_promote_template(self, template_name: str) -> bool:
        """Verifica se um template deve ser promovido de rascunho para validado.

        Critérios de promoção:
        - Mínimo de 10 extrações aprovadas
        - Taxa de sucesso ≥ 90%
        - Mínimo de 5 aprovações consecutivas sem rejeição

        Args:
            template_name: Nome do template.

        Returns:
            True se o template deve ser promovido.
        """
        stats = self.get_template_stats(template_name)

        approved = stats.get("approved", 0)
        success_rate = stats.get("success_rate", 0)
        consecutive = stats.get("consecutive_successes", 0)

        return approved >= 10 and success_rate >= 0.90 and consecutive >= 5

    def should_demote_template(self, template_name: str) -> bool:
        """Verifica se um template validado deve ser rebaixado.

        Critérios de rebaixamento:
        - 3+ rejeições consecutivas

        Args:
            template_name: Nome do template.

        Returns:
            True se o template deve ser rebaixado.
        """
        stats = self.get_template_stats(template_name)
        return stats.get("consecutive_failures", 0) >= 3
