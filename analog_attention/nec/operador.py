"""Operador Θ — Combinador de Estados.

f(EE, EA, DA) = Σ w[d] × estado_d
x = (1 − sim)
output → campo P

Combina a Estrutura Emocional (EE), Estado Atual (EA)
e Delta Associativo (DA) num único campo de probabilidade
que orienta a projeção de possíveis.
"""
import numpy as np
from .emocional import EstruturaEmocional
from .estado import EstadoAtual
from .associativo import ReferenciaAssociativa


# Pesos do Operador Θ por dimensão
# [valencia, intensidade, coerencia, ativacao, carga, novidade, salto]
_PESOS_THETA = np.array([0.20, 0.25, 0.15, 0.20, 0.10, 0.25, 0.30])


def operador_theta(
    ee: EstruturaEmocional,
    ea: EstadoAtual,
    ra: ReferenciaAssociativa,
) -> np.ndarray:
    """Calcula o campo P combinando os três estados.

    Fórmula:
        estado_d = [ee.vetor | delta_novidade | flag_salto]
        campo_P  = Σ w[d] × estado_d
        x        = 1 − similaridade_ancora  (modulador de abertura)
        campo_P  *= (1 + x)  — saltos ampliam o campo

    Returns:
        campo_P: vetor de probabilidade normalizado [0, 1]
    """
    # Vetor de estado combinado
    estado_d = np.array([
        (ee.valencia + 1) / 2,  # normalizado para [0,1]
        ee.intensidade,
        ee.coerencia,
        ee.ativacao,
        ee.carga,
        ra.delta_novidade,
        float(ea.flag_salto),
    ])

    # Produto ponderado
    campo_p = _PESOS_THETA * estado_d

    # Modulador x = 1 − similaridade (saltos abrem o campo)
    x = 1.0 - ea.similaridade_ancora
    campo_p *= (1.0 + x)

    # Normaliza para [0, 1]
    total = campo_p.sum()
    if total > 0:
        campo_p /= total

    return campo_p
