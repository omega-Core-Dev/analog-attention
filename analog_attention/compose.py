import numpy as np
from .token import Token


def compose(tokens: list[Token]) -> dict:
    """Agrega sinais do cluster de tokens ativados.

    Returns:
        {
            "cluster":         list[Token] — tokens ativados,
            "pesos":           dict[str, float] — peso normalizado,
            "sinal_total":     float,
            "assinatura_cluster": np.ndarray — média ponderada das assinaturas,
        }
    """
    cluster = [t for t in tokens if t.foi_atingido]
    sinal_total = sum(t.sinal_recebido for t in cluster)

    if sinal_total == 0 or not cluster:
        return {
            "cluster": [],
            "pesos": {},
            "sinal_total": 0.0,
            "assinatura_cluster": np.zeros_like(tokens[0].assinatura) if tokens else np.array([]),
        }

    pesos = {t.nome: t.sinal_recebido / sinal_total for t in cluster}
    assinatura_cluster = sum(pesos[t.nome] * t.assinatura for t in cluster)

    return {
        "cluster": cluster,
        "pesos": pesos,
        "sinal_total": sinal_total,
        "assinatura_cluster": assinatura_cluster,
    }
