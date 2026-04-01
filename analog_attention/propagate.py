import math
from .token import Token
from .scan import scan
from .broadcast import broadcast
from .candidate import candidate


def _distancia(a: Token, b: Token) -> float:
    x1, y1 = a.coordenada
    x2, y2 = b.coordenada
    return math.hypot(x2 - x1, y2 - y1)


def propagate(
    tokens: list[Token],
    intensidade_professor: float = 5.0,
    ruido: float = 0.0,
    falloff: str = "sqrt",
    fator_social: float = 0.5,
    threshold: float = 0.4,
    raio: float | None = None,
    d_max: float = 10.0,
) -> list[Token]:
    """Pipeline completo: scan → broadcast → candidate → propagação social.

    Fases:
        1. scan()      O(n)     — constrói mapa de coordenadas
        2. broadcast() O(n)     — professor envia sinal para cada token
        3. candidate() O(n)     — filtra tokens elegíveis
        4. propagate   O(k·r)   — candidatos repassam sinal a vizinhos no raio

    Args:
        tokens: Lista de tokens com coordenadas definidas.
        intensidade_professor: Força do sinal inicial (I).
        ruido: Ruído aplicado ao sinal do professor (η).
        falloff: Tipo de decaimento do broadcast ('quadratic' | 'sqrt' | 'linear').
        fator_social: Fator de atenuação na propagação social (φ ∈ [0, 1]).
        threshold: Sinal mínimo para um token se tornar candidato (θ).
        raio: Raio máximo de propagação social. None = sem limite.
        d_max: Distância máxima de referência para falloffs 'sqrt' e 'linear'.

    Returns:
        Lista de tokens com sinal_recebido atualizado.

    Complexidade: O(n) + O(k·r), onde k ≤ n e r ≤ n.
    """
    # Fase 1 — scan: constrói mapa de coordenadas (pago uma única vez)
    _coordinate_map = scan(tokens)  # noqa: F841 — disponível para extensões futuras

    # Fase 2 — broadcast: professor emite sinal para cada token
    for token in tokens:
        sinal = broadcast(intensidade_professor, token, ruido=ruido, falloff=falloff, d_max=d_max)
        token.receber_sinal(sinal)

    # Fase 3 — candidate: filtra tokens elegíveis para propagar
    candidatos = [t for t in tokens if candidate(t, threshold=threshold)]

    # Fase 4 — propagação social: candidatos repassam sinal a vizinhos no raio
    for origem in candidatos:
        for vizinho in tokens:
            if vizinho is origem:
                continue
            d = _distancia(origem, vizinho)
            if raio is not None and d > raio:
                continue
            sinal_social = origem.sinal_recebido * fator_social / (d + 1)
            vizinho.receber_sinal(max(sinal_social, 0.0))

    return tokens
