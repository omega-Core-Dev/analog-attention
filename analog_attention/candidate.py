from .token import Token


def candidate(token: Token, threshold: float = 0.3) -> bool:
    """Retorna True se o token está elegível para propagar na cadeia.

    Args:
        token:     token a avaliar.
        threshold: sinal mínimo para elegibilidade (θ).

    Returns:
        True se token.sinal_recebido >= threshold.
    """
    return token.sinal_recebido >= threshold
