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
