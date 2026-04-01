from .token import Token


def compose(tokens: list[Token]) -> dict:
    """Agrega sinais do cluster de tokens atingidos.

    Produz um dicionário com os sinais de todos os tokens
    que foram atingidos (sinal_recebido > 0), normalizado
    pelo sinal total do cluster.

    Args:
        tokens: Lista de tokens após propagação.

    Returns:
        {
            "cluster": list[Token] — tokens atingidos,
            "pesos": dict[str, float] — peso normalizado por token,
            "sinal_total": float — soma dos sinais do cluster,
        }

    Complexidade: O(k) onde k = |cluster|.
    """
    cluster = [t for t in tokens if t.foi_atingido]
    sinal_total = sum(t.sinal_recebido for t in cluster)

    if sinal_total == 0:
        pesos = {t.nome: 0.0 for t in cluster}
    else:
        pesos = {t.nome: t.sinal_recebido / sinal_total for t in cluster}

    return {
        "cluster": cluster,
        "pesos": pesos,
        "sinal_total": sinal_total,
    }
