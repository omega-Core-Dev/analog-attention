import math
from .token import Token

_FALLOFF_OPTIONS = ("quadratic", "sqrt", "linear")


def broadcast(
    intensidade: float,
    token: Token,
    ruido: float = 0.0,
    falloff: str = "sqrt",
    d_max: float = 10.0,
) -> float:
    """Professor envia sinal para um token com falloff por distância.

    Args:
        intensidade: Força do sinal emitido (I).
        token: Token receptor.
        ruido: Ruído subtraído do sinal resultante (η).
        falloff: Tipo de decaimento — 'quadratic' | 'sqrt' | 'linear'.
        d_max: Distância máxima de referência (usada por 'sqrt' e 'linear').

    Returns:
        Sinal calculado (≥ 0).

    Complexidade: O(1) por token.
    """
    if falloff not in _FALLOFF_OPTIONS:
        raise ValueError(f"falloff deve ser um de {_FALLOFF_OPTIONS}, recebido: {falloff!r}")

    x, y = token.coordenada
    d = math.hypot(x, y)

    if falloff == "quadratic":
        sinal = intensidade / (d ** 2 + 1) - ruido
    elif falloff == "sqrt":
        t = min(d / d_max, 1.0) if d_max > 0 else 1.0
        sinal = intensidade * (1 - math.sqrt(t) * 0.75) - ruido
    else:  # linear
        t = min(d / d_max, 1.0) if d_max > 0 else 1.0
        sinal = intensidade * (1 - t) - ruido

    return max(sinal, 0.0)


def broadcast_all(
    tokens: list[Token],
    intensidade: float,
    ruido: float = 0.0,
    falloff: str = "sqrt",
    d_max: float = 10.0,
) -> list[float]:
    """Emite sinal do professor para todos os tokens em batch.

    Args:
        tokens: Lista de tokens receptores.
        intensidade: Força do sinal emitido (I).
        ruido: Ruído subtraído do sinal resultante (η).
        falloff: Tipo de decaimento — 'quadratic' | 'sqrt' | 'linear'.
        d_max: Distância máxima de referência (usada por 'sqrt' e 'linear').

    Returns:
        Lista de sinais calculados, na mesma ordem dos tokens.

    Complexidade: O(n).
    """
    return [broadcast(intensidade, t, ruido=ruido, falloff=falloff, d_max=d_max) for t in tokens]
