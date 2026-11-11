import re
import json
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class Reviewer:
    """Valida extrações de COC com regras automáticas.

    Executa validações estruturais nos dados extraídos (formato de código,
    data válida, campos obrigatórios) e determina se a extração deve ser
    aprovada automaticamente ou encaminhada para revisão humana.
    """

    # -----------------------------------------------------------------
    # REGRAS DE VALIDAÇÃO
    # -----------------------------------------------------------------
    def _validate_codigo(self, codigo: str) -> Tuple[bool, str]:
        """Valida o formato do código da amostra.

        Esperado: exatamente 7 dígitos numéricos.

        Returns:
            Tupla (é_válido, mensagem).
        """
        if not codigo:
            return True, "Código vazio (aceito se houver identificacao)"

        codigo_limpo = codigo.strip()
        if re.match(r"^\d{7}$", codigo_limpo):
            return True, "OK"

        return False, f"Código '{codigo_limpo}' não tem 7 dígitos numéricos"

    def _validate_data(self, data: str) -> Tuple[bool, str]:
        """Valida o formato e valor da data.

        Esperado: DD/MM/AAAA com data real.

        Returns:
            Tupla (é_válido, mensagem).
        """
        if not data:
            return True, "Data vazia (aceita)"

        data_limpa = data.strip()

        # Verifica formato DD/MM/AAAA ou DD/MM/AA
        match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", data_limpa)
        if not match:
            return False, f"Data '{data_limpa}' não está no formato DD/MM/AAAA"

        dia, mes, ano = int(match.group(1)), int(match.group(2)), int(match.group(3))

        # Normaliza ano de 2 dígitos para 4 dígitos (ex: 26 -> 2026)
        if ano < 100:
            ano += 2000

        # Validação de data real
        try:
            datetime(ano, mes, dia)
        except ValueError:
            return False, f"Data '{data_limpa}' não é uma data válida"

        # Check de ano razoável (entre 2020 e ano atual + 1)
        ano_atual = datetime.now().year
        if ano < 2020 or ano > ano_atual + 1:
            return False, f"Ano {ano} está fora do intervalo esperado (2020-{ano_atual + 1})"

        return True, "OK"

    def _validate_hora(self, hora: str) -> Tuple[bool, str]:
        """Valida o formato da hora.

        Esperado: HH:MM (00:00 a 23:59).

        Returns:
            Tupla (é_válido, mensagem).
        """
        if not hora:
            return True, "Hora vazia (aceita)"

        hora_limpa = hora.strip()

        match = re.match(r"^(\d{1,2}):(\d{2})$", hora_limpa)
        if not match:
            return False, f"Hora '{hora_limpa}' não está no formato HH:MM"

        h, m = int(match.group(1)), int(match.group(2))
        if h > 23 or m > 59:
            return False, f"Hora '{hora_limpa}' está fora do intervalo válido"

        return True, "OK"

    def _validate_identificacao(self, identificacao: str) -> Tuple[bool, str]:
        """Valida a identificação da amostra.

        Regras:
        - Não pode ser só espaços ou pontuação
        - Tamanho razoável (≤ 200 chars)

        Returns:
            Tupla (é_válido, mensagem).
        """
        if not identificacao:
            return True, "Identificação vazia (aceita se houver código)"

        id_limpa = identificacao.strip()

        if len(id_limpa) > 200:
            return False, f"Identificação muito longa ({len(id_limpa)} chars)"

        # Verifica se tem conteúdo real (não só pontuação/espaços)
        if not re.search(r"[a-zA-Z0-9]", id_limpa):
            return False, f"Identificação '{id_limpa[:50]}' não contém texto válido"

        return True, "OK"

    # -----------------------------------------------------------------
    # REVISÃO DE UMA AMOSTRA
    # -----------------------------------------------------------------
    def review_sample(self, sample: Dict[str, str]) -> Dict[str, Any]:
        """Revisa uma amostra individual extraída.

        Args:
            sample: Dicionário com campos (template, codigo, identificacao, data, hora).

        Returns:
            Dicionário com resultado da revisão incluindo:
            - approved: bool
            - issues: lista de problemas encontrados
            - warnings: lista de avisos não-bloqueantes
        """
        issues = []
        warnings = []

        codigo = sample.get("codigo", "")
        identificacao = sample.get("identificacao", "")
        data = sample.get("data", "")
        hora = sample.get("hora", "")

        # Validação obrigatória: precisa ter código OU identificação
        if not codigo and not identificacao:
            issues.append("Amostra sem código E sem identificação — dados insuficientes")

        # Validar código
        ok, msg = self._validate_codigo(codigo)
        if not ok:
            issues.append(msg)

        # Validar data
        ok, msg = self._validate_data(data)
        if not ok:
            issues.append(msg)
        else:
            # Se a data foi validada (e possivelmente normalizada p/ 4 dígitos),
            # precisamos garantir que o valor final no sample reflita isso.
            match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", data.strip())
            if match:
                d, m, a = int(match.group(1)), int(match.group(2)), int(match.group(3))
                if a < 100:
                    a += 2000
                normalized = f"{str(d).zfill(2)}/{str(m).zfill(2)}/{str(a).zfill(4)}"
                sample["data"] = normalized

        # Validar hora
        ok, msg = self._validate_hora(hora)
        if not ok:
            warnings.append(msg)  # Hora inválida é warning, não bloqueante

        # Validar identificação
        ok, msg = self._validate_identificacao(identificacao)
        if not ok:
            warnings.append(msg)

        # Warnings adicionais
        if not data:
            warnings.append("Data não preenchida")
        if not hora:
            warnings.append("Hora não preenchida")
        if codigo and not identificacao:
            warnings.append("Código presente mas sem identificação")

        approved = len(issues) == 0

        return {
            "approved": approved,
            "issues": issues,
            "warnings": warnings,
            "sample": sample,
        }

    # -----------------------------------------------------------------
    # REVISÃO DE UM LOTE COMPLETO
    # -----------------------------------------------------------------
    def review_extraction(self, extraction_result: str) -> Dict[str, Any]:
        """Revisa todas as amostras de uma extração.

        Args:
            extraction_result: JSON string do resultado do FillerAgentTool.

        Returns:
            Dicionário com resultado da revisão do lote:
            - all_approved: bool (se TODAS as amostras passaram)
            - total_samples: int
            - approved_count: int
            - rejected_count: int
            - results: lista de revisões individuais
        """
        try:
            result = json.loads(extraction_result)
        except json.JSONDecodeError:
            return {
                "all_approved": False,
                "total_samples": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "error": "JSON de extração inválido",
                "results": [],
            }

        samples = result.get("samples", [])
        template = result.get("template", "DESCONHECIDO")

        if not samples:
            return {
                "all_approved": False,
                "total_samples": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "template": template,
                "results": [],
            }

        reviews = []
        approved_count = 0
        rejected_count = 0

        for sample in samples:
            review = self.review_sample(sample)
            reviews.append(review)
            if review["approved"]:
                approved_count += 1
            else:
                rejected_count += 1

        all_approved = rejected_count == 0

        # Agrega todos os motivos de rejeição de todas as amostras
        all_rejection_reasons = []
        for r in reviews:
            if not r["approved"]:
                all_rejection_reasons.extend(r["issues"])

        # Remove duplicatas e junta em uma string
        rejection_summary = "; ".join(list(dict.fromkeys(all_rejection_reasons)))

        log_fn = logger.info if all_approved else logger.warning
        log_fn(
            f"📋 Revisão [{template}]: {approved_count}/{len(samples)} aprovadas "
            f"| {rejected_count} rejeitadas | Motivos: {rejection_summary}"
        )

        return {
            "all_approved": all_approved,
            "total_samples": len(samples),
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "template": template,
            "results": reviews,
            "rejection_summary": rejection_summary,
        }
