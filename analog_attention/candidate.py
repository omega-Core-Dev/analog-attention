from .token import Token


def candidate(token: Token, threshold: float = 0.4) -> bool:
    """Retorna True se o token está elegível para propagar.

    Um token é candidato quando seu sinal acumulado atinge o threshold,
    indicando que recebeu atenção suficiente para repassar o sinal.

    Args:
        token: Token a avaliar.
        threshold: Sinal mínimo para elegibilidade (θ).

    Returns:
        True se token.sinal_recebido >= threshold.

    Complexidade: O(1).
    """
    return token.sinal_recebido >= threshold
