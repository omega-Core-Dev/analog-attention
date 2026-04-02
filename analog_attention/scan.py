import numpy as np
from .token import Token


def scan(tokens: list[Token]) -> dict[str, np.ndarray]:
    """Constrói índice de assinaturas de frequência — O(n).

    Returns:
        { token.nome: token.assinatura } para todos os tokens.
    """
    return {t.nome: t.assinatura for t in tokens}
