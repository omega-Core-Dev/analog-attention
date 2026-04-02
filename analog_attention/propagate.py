"""Propagação em cadeia harmônica.

Candidatos repassam sinal ao vizinho de maior ressonância —
formando uma corrente ponto a ponto, não uma onda radial.

Complexidade: O(k · n) no pior caso, O(k · hops) típico.
"""
import warnings
import numpy as np
from .token import Token
from .broadcast import broadcast
from .candidate import candidate
from .signature import correlacao

_LIMIAR_WARNING = 50


def _vizinho_mais_harmonico(origem: Token, tokens: list[Token], visitados: set) -> Token | None:
    """Retorna o token não visitado com maior ressonância harmônica com a origem."""
    melhor, melhor_score = None, -1.0
    for t in tokens:
        if t is origem or t.nome in visitados:
            continue
        score = correlacao(origem.assinatura, t.assinatura)
        if score > melhor_score:
            melhor_score = score
            melhor = t
    return melhor


def propagate(
    tokens: list[Token],
    query: np.ndarray,
    intensidade: float = 5.0,
    curvatura: float = 4.0,
    ruido: float = 0.0,
    fator_social: float = 0.5,
    threshold: float = 0.3,
    max_hops: int = 3,
) -> list[Token]:
    """Pipeline: broadcast parabólico → candidatos → cadeia harmônica.

    Fases:
        1. broadcast  O(n)       — professor ativa tokens por ressonância parabólica
        2. candidate  O(n)       — filtra elegíveis acima do threshold
        3. cadeia     O(k·hops)  — cada candidato passa sinal ao vizinho mais harmônico

    Args:
        tokens:       tokens do input com assinatura BASA.
        query:        assinatura de frequência do professor (input completo).
        intensidade:  amplitude do broadcast (I).
        curvatura:    curvatura parabólica (κ).
        ruido:        ruído do broadcast (η).
        fator_social: atenuação por hop na cadeia (φ).
        threshold:    limiar mínimo para ser candidato.
        max_hops:     comprimento máximo da cadeia por candidato.

    Returns:
        Tokens com sinal_recebido atualizado.
    """
    if len(tokens) > _LIMIAR_WARNING:
        warnings.warn(
            f"propagate() com {len(tokens)} tokens — considere segmentar o input.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Fase 1 — broadcast parabólico
    for token in tokens:
        sinal = broadcast(query, token, intensidade, curvatura, ruido)
        token.receber_sinal(sinal)

    # Fase 2 — candidatos
    candidatos = [t for t in tokens if candidate(t, threshold)]

    # Fase 3 — cadeia harmônica
    for origem in candidatos:
        visitados = {origem.nome}
        atual = origem
        sinal_atual = atual.sinal_recebido

        for _ in range(max_hops):
            proximo = _vizinho_mais_harmonico(atual, tokens, visitados)
            if proximo is None:
                break
            sinal_atual *= fator_social
            proximo.receber_sinal(sinal_atual)
            visitados.add(proximo.nome)
            atual = proximo

    return tokens
