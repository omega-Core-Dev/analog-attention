"""Broadcast parabólico por frequência.

O professor emite na frequência dominante da query.
A ativação de cada token decai parabolicamente pela distância espectral.

σ(f) = I · max(0, 1 − κ · (f_token − f_peak)²) − η

Onde:
    I     = intensidade
    κ     = curvatura da parábola (controla largura do feixe)
    f_peak = frequência dominante da query (professor)
    η     = ruído
"""
import numpy as np
from .token import Token
from .signature import correlacao, frequencia_dominante


def broadcast(
    query: np.ndarray,
    token: Token,
    intensidade: float = 5.0,
    curvatura: float = 4.0,
    ruido: float = 0.0,
) -> float:
    """Ativação parabólica de um token pela query do professor.

    Args:
        query:      assinatura de frequência do professor (vetor FFT).
        token:      token receptor.
        intensidade: amplitude máxima do sinal (I).
        curvatura:  quão rápido o sinal cai fora do pico (κ).
                    Alto κ = feixe estreito. Baixo κ = feixe largo.
        ruido:      ruído subtraído do resultado (η).

    Returns:
        Sinal de ativação ≥ 0.

    Complexidade: O(1) por token.
    """
    f_peak = frequencia_dominante(query)
    f_token = token.frequencia_dominante

    sinal = intensidade * max(0.0, 1.0 - curvatura * (f_token - f_peak) ** 2)
    return max(sinal - ruido, 0.0)


def broadcast_all(
    query: np.ndarray,
    tokens: list[Token],
    intensidade: float = 5.0,
    curvatura: float = 4.0,
    ruido: float = 0.0,
) -> list[float]:
    """Broadcast parabólico para todos os tokens. O(n).

    Returns:
        Lista de sinais na mesma ordem dos tokens.
    """
    return [broadcast(query, t, intensidade, curvatura, ruido) for t in tokens]
