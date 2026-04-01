"""analog-attention — Propagação de Atenção por Coordenadas de Token.

Mecanismo de atenção baseado em endereçamento direto por coordenadas,
como alternativa à atenção O(n²) do Transformer clássico.

Pipeline: scan → broadcast → candidate → propagate → compose

Complexidade: O(n) + O(k) + O(k·r)  onde k, r << n
"""

from .token import Token
from .scan import scan
from .broadcast import broadcast, broadcast_all
from .candidate import candidate
from .propagate import propagate
from .compose import compose

__version__ = "0.2.0"
__author__ = "Zaqueu Ribeiro da Costa"
__license__ = "GPL-3.0"

__all__ = [
    "Token",
    "scan",
    "broadcast",
    "broadcast_all",
    "candidate",
    "propagate",
    "compose",
]
