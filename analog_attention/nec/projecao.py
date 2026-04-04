"""Estágio 04 — Projeção de Possíveis.

Usa o campo P (Operador Θ) para determinar o modo de resposta
e selecionar os tokens mais relevantes para a geração.

f(estado_H | campo_P) → distribuição sobre modos de resposta
"""
import numpy as np
from dataclasses import dataclass
from ..token import Token
from .emocional import EstruturaEmocional
from .estado import EstadoAtual
from .associativo import ReferenciaAssociativa


MODOS_RESPOSTA = [
    "exploratório",   # alta novidade, alta ativação
    "confirmatório",  # baixa novidade, alta coerência
    "adaptativo",     # salto de contexto detectado
    "tenso",          # alta intensidade, valencia negativa
    "estável",        # baixa variância, fluxo contínuo
]


@dataclass
class ProjecaoPossiveis:
    modo: str                    # modo de resposta dominante
    distribuicao: dict           # probabilidade por modo
    tokens_chave: list[Token]    # tokens mais relevantes para geração
    campo_p: np.ndarray          # campo de probabilidade (Operador Θ)
    confianca: float             # confiança na projeção [0, 1]

    def __repr__(self):
        return (
            f"PP(modo={self.modo} "
            f"confianca={self.confianca:.3f} "
            f"tokens_chave={[t.nome for t in self.tokens_chave[:3]]})"
        )


def calcular_projecao(
    tokens: list[Token],
    ee: EstruturaEmocional,
    ea: EstadoAtual,
    ra: ReferenciaAssociativa,
    campo_p: np.ndarray,
    top_k: int = 5,
) -> ProjecaoPossiveis:
    """Projeta o campo de possíveis e seleciona modo + tokens chave.

    Lógica de seleção de modo:
        adaptativo   → flag_salto = True (prioridade máxima)
        tenso        → intensidade > 0.6 AND valencia < -0.1
        exploratório → delta_novidade > 0.5 AND ativacao > 0.4
        confirmatório → delta_novidade < 0.3 AND coerencia > 0.6
        estável      → default
    """
    # Seleção do modo baseada no estado
    if ea.flag_salto:
        modo = "adaptativo"
    elif ee.intensidade > 0.6 and ee.valencia < -0.1:
        modo = "tenso"
    elif ra.delta_novidade > 0.5 and ee.ativacao > 0.4:
        modo = "exploratório"
    elif ra.delta_novidade < 0.3 and ee.coerencia > 0.6:
        modo = "confirmatório"
    else:
        modo = "estável"

    # Distribuição de probabilidade por modo
    scores = {
        "exploratório":  ra.delta_novidade * ee.ativacao,
        "confirmatório": (1 - ra.delta_novidade) * ee.coerencia,
        "adaptativo":    float(ea.flag_salto) * ea.intensidade_salto,
        "tenso":         ee.intensidade * max(0, -ee.valencia),
        "estável":       (1 - ee.intensidade) * ee.coerencia,
    }
    total = sum(scores.values()) or 1.0
    distribuicao = {k: round(v / total, 4) for k, v in scores.items()}

    # Tokens chave: top_k por sinal recebido
    tokens_ord = sorted(tokens, key=lambda t: t.sinal_recebido, reverse=True)
    tokens_chave = tokens_ord[:top_k]

    # Confiança: quão dominante é o modo escolhido
    confianca = float(distribuicao[modo])

    return ProjecaoPossiveis(
        modo=modo,
        distribuicao=distribuicao,
        tokens_chave=tokens_chave,
        campo_p=campo_p,
        confianca=confianca,
    )
