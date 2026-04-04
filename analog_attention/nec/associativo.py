"""Estágio 03 — Referência Associativa.

Calcula o delta de novidade e classifica tokens em:
    D1 (familiar_novo):      já vi esse padrão, mas em contexto novo
    D2 (familiar_conhecido): já vi esse padrão, contexto reconhecido

O delta de novidade é o coração da memória como transformação —
não "o que lembro" mas "o quanto isso difere do que me formou".
"""
import numpy as np
from dataclasses import dataclass, field
from ..token import Token
from ..output import AncoradeContexto
from ..signature import correlacao
from .estado import EstadoAtual


@dataclass
class ReferenciaAssociativa:
    delta_novidade: float        # D ∈ [0, 1] — 0 = totalmente familiar
    D1: list[Token]              # familiar_novo — padrão conhecido, contexto novo
    D2: list[Token]              # familiar_conhecido — padrão e contexto conhecidos
    tokens_ineditos: list[Token] # sem referência anterior
    peso_associativo: float      # peso geral da referência [0, 1]

    def __repr__(self):
        return (
            f"RA(delta={self.delta_novidade:.3f} "
            f"D1={len(self.D1)} D2={len(self.D2)} "
            f"inéditos={len(self.tokens_ineditos)} "
            f"peso={self.peso_associativo:.3f})"
        )


_LIMIAR_D1 = 0.4   # correlação mínima para "familiar"
_LIMIAR_D2 = 0.65  # correlação mínima para "bem conhecido"


def calcular_referencia_associativa(
    tokens: list[Token],
    ea: EstadoAtual,
    ancora: AncoradeContexto | None,
) -> ReferenciaAssociativa:
    """Classifica tokens por grau de familiaridade com o estado anterior.

    Delta de novidade:
        D = 1 − correlação(assinatura_token, assinatura_ancora)

    Classificação:
        D1 (familiar_novo):      flag_salto=True  + correlação média
        D2 (familiar_conhecido): flag_salto=False + correlação alta

    Args:
        tokens: tokens do input atual com sinal processado.
        ea:     Estado Atual (Stage 02).
        ancora: âncora do ciclo anterior.
    """
    if ancora is None or ancora.assinatura.sum() == 0:
        # Primeiro ciclo — tudo é inédito
        return ReferenciaAssociativa(
            delta_novidade=1.0,
            D1=[],
            D2=[],
            tokens_ineditos=list(tokens),
            peso_associativo=0.0,
        )

    ancora_sig = ancora.assinatura

    correlacoes = []
    for t in tokens:
        n = min(len(t.assinatura), len(ancora_sig))
        c = correlacao(t.assinatura[:n], ancora_sig[:n])
        correlacoes.append(c)

    delta_novidade = float(np.clip(1.0 - np.mean(correlacoes), 0.0, 1.0))

    D1, D2, ineditos = [], [], []
    for t, c in zip(tokens, correlacoes):
        if c >= _LIMIAR_D2 and not ea.flag_salto:
            D2.append(t)
        elif c >= _LIMIAR_D1:
            D1.append(t)
        else:
            ineditos.append(t)

    peso_associativo = float(np.mean(correlacoes))

    return ReferenciaAssociativa(
        delta_novidade=delta_novidade,
        D1=D1,
        D2=D2,
        tokens_ineditos=ineditos,
        peso_associativo=peso_associativo,
    )
