"""Estágio 02 — Estado Atual.

Lê a estrutura emocional e detecta saltos de contexto
comparando com o estado anterior (âncora do ciclo N-1).

O detector de salto é o que dá à entidade consciência
de quando o campo mudou de natureza — não só de conteúdo.
"""
import numpy as np
from dataclasses import dataclass
from .emocional import EstruturaEmocional
from ..output import AncoradeContexto
from ..signature import correlacao


THRESHOLD_SALTO = 0.35   # similaridade abaixo disso = salto de contexto


@dataclass
class EstadoAtual:
    vetor_estado: np.ndarray     # vetor emocional do ciclo atual
    similaridade_ancora: float   # quão próximo do ciclo anterior
    flag_salto: bool             # True = mudança de contexto detectada
    intensidade_salto: float     # magnitude da mudança [0, 1]
    modo: str                    # 'contínuo' | 'salto' | 'primeiro_ciclo'

    def __repr__(self):
        salto = f"SALTO({self.intensidade_salto:.3f})" if self.flag_salto else "contínuo"
        return (
            f"EA(modo={self.modo} "
            f"sim_ancora={self.similaridade_ancora:.3f} "
            f"estado={salto})"
        )


def calcular_estado_atual(
    ee: EstruturaEmocional,
    ancora: AncoradeContexto | None,
) -> EstadoAtual:
    """Calcula o estado atual e detecta saltos de contexto.

    Args:
        ee:     Estrutura Emocional do ciclo atual (Stage 01).
        ancora: Âncora do ciclo anterior (None = primeiro ciclo).

    Returns:
        EstadoAtual com flag de salto e modo de operação.
    """
    vetor = ee.vetor

    if ancora is None:
        return EstadoAtual(
            vetor_estado=vetor,
            similaridade_ancora=1.0,
            flag_salto=False,
            intensidade_salto=0.0,
            modo="primeiro_ciclo",
        )

    # Compara vetor emocional atual com assinatura da âncora anterior
    sim = correlacao(vetor, ancora.assinatura[:len(vetor)])
    intensidade_salto = float(np.clip(1.0 - sim, 0.0, 1.0))
    flag_salto = sim < THRESHOLD_SALTO

    modo = "salto" if flag_salto else "contínuo"

    return EstadoAtual(
        vetor_estado=vetor,
        similaridade_ancora=float(sim),
        flag_salto=flag_salto,
        intensidade_salto=intensidade_salto,
        modo=modo,
    )
