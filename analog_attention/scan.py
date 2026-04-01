from .token import Token


def scan(tokens: list[Token]) -> dict[str, tuple[float, float]]:
    """Constrói índice de coordenadas — pago uma única vez.

    Complexidade: O(n)

    Returns:
        { token.nome: token.coordenada } para todos os tokens.
    """
    return {token.nome: token.coordenada for token in tokens}
