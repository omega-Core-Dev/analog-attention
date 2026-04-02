import numpy as np

N_COMPONENTS = 16  # FFT components kept per token


def compute_signature(texto: str, n: int = N_COMPONENTS) -> tuple[np.ndarray, np.ndarray]:
    """Converte texto em assinatura de frequência.

    Espectro = histograma de caracteres mapeados em N bins de frequência.
    Cada caractere contribui para o bin correspondente ao seu código ASCII % N.
    Isso distribui tokens de forma determinística pelo espaço de frequências.

    Returns:
        magnitude: vetor de energia por bin de frequência (normalizado, soma=1)
        fase:      vetor de fase por bin (posição ponderada dos caracteres)
    """
    if not texto:
        return np.zeros(n), np.zeros(n)

    magnitude = np.zeros(n)
    fase = np.zeros(n)

    for i, c in enumerate(texto):
        code = ord(c) % 128
        bin_idx = code % n
        magnitude[bin_idx] += 1.0
        # Fase: posição do caractere no token (0 a 2π)
        fase[bin_idx] += (i / max(len(texto) - 1, 1)) * 2 * np.pi

    # Normaliza magnitude (distribuição de energia)
    total = magnitude.sum()
    if total > 0:
        magnitude /= total

    # Normaliza fase para [-π, π]
    fase = (fase % (2 * np.pi)) - np.pi

    return magnitude, fase


def frequencia_dominante(magnitude: np.ndarray) -> float:
    """Centroide espectral normalizado [0, 1].

    Usa média ponderada dos índices de frequência (excluindo DC)
    para distribuir tokens pelo espaço de classes de onda.
    """
    n = len(magnitude)
    if n == 0:
        return 0.0
    # Suprime componente DC (índice 0) — evita colapso de todos em ONDA_INFRA
    m = magnitude.copy()
    m[0] = 0.0
    total = m.sum()
    if total == 0:
        # Fallback: usa byte médio do sinal para diferenciação
        return 0.0
    indices = np.arange(n, dtype=float)
    centroide = float(np.dot(indices, m) / total)
    return float(np.clip(centroide / (n - 1), 0.0, 1.0))


def estabilidade(magnitude: np.ndarray) -> float:
    """Estabilidade harmônica [0, 1].

    Alta estabilidade = energia concentrada em poucos componentes (sinal limpo).
    Baixa estabilidade = energia espalhada (sinal ruidoso).
    """
    if magnitude.sum() == 0:
        return 0.0
    p = magnitude / (magnitude.sum() + 1e-10)
    entropia = -np.sum(p * np.log(p + 1e-10))
    entropia_max = np.log(len(magnitude))
    return float(1.0 - entropia / entropia_max) if entropia_max > 0 else 0.0


def correlacao(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade cosseno entre duas assinaturas [0, 1]."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))
