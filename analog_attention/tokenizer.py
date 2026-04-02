"""Tokenizador customizado — sem estrutura linguística.

Cada token carrega sua assinatura de frequência (via BASA),
classe de onda, pertencimento e métricas de sinal.
"""
import re
import numpy as np
from .token import Token
from .signature import compute_signature, frequencia_dominante, estabilidade
from .basa import classificar


def tokenizar(texto: str) -> list[Token]:
    """Converte texto em lista de Tokens com assinatura de frequência.

    Estratégia de segmentação:
        - Palavras e contrações como unidades
        - Pontuação como tokens separados
        - Sem normalização linguística

    Pertencimento:
        Todos os tokens do mesmo input se conhecem — compartilham contexto.

    Returns:
        Lista de Tokens prontos para atenção.
    """
    if not texto.strip():
        return []

    fragmentos = re.findall(r"\w+(?:'\w+)*|[^\w\s]", texto)

    tokens: list[Token] = []
    for frag in fragmentos:
        mag, fase = compute_signature(frag)
        freq = frequencia_dominante(mag)
        estab = estabilidade(mag)
        classe, subclasse = classificar(freq)

        t = Token(
            nome=frag,
            assinatura=mag,
            fase=fase,
            classe=classe,
            subclasse=subclasse,
            magnitude_total=float(mag.sum()),
            estabilidade=estab,
        )
        tokens.append(t)

    # Pertencimento: cada token conhece os demais do mesmo input
    todos = [t.nome for t in tokens]
    for t in tokens:
        t.pertence_a = [n for n in todos if n != t.nome]

    return tokens


def assinatura_do_texto(texto: str) -> np.ndarray:
    """Assinatura FFT do texto completo — usada como query do professor."""
    mag, _ = compute_signature(texto)
    return mag
