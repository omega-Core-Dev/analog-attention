"""analog-attention — Propagação de Atenção por Coordenadas de Frequência.

Pipeline: tokenizar → broadcast parabólico → cadeia harmônica → frequency log

Complexidade: O(n) broadcast + O(k·hops) cadeia
"""

from .token import Token
from .signature import compute_signature, frequencia_dominante, estabilidade, correlacao
from .basa import classificar, WAVE_CLASSES
from .tokenizer import tokenizar, assinatura_do_texto
from .scan import scan
from .broadcast import broadcast, broadcast_all
from .candidate import candidate
from .propagate import propagate
from .compose import compose
from .output import gerar_log, imprimir_log, limiar_parabolico, AncoradeContexto, FrequencyLog
from .pipeline import processar

__version__ = "0.3.0"
__author__ = "Zaqueu Ribeiro da Costa"
__license__ = "GPL-3.0"

__all__ = [
    "Token",
    "compute_signature", "frequencia_dominante", "estabilidade", "correlacao",
    "classificar", "WAVE_CLASSES",
    "tokenizar", "assinatura_do_texto",
    "scan",
    "broadcast", "broadcast_all",
    "candidate",
    "propagate",
    "compose",
    "gerar_log", "imprimir_log", "limiar_parabolico",
    "AncoradeContexto", "FrequencyLog",
    "processar",
]
