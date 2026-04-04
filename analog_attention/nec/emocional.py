"""Estágio 01 — Estrutura Emocional.

Lê os tokens com assinatura de frequência e extrai
o vetor de estado emocional por âmbito.

Âmbitos (dimensões do vetor emocional):
    valencia    [-1, 1]  negativo ↔ positivo
    intensidade [0, 1]   calmo ↔ intenso
    coerencia   [0, 1]   conflitado ↔ coerente
    ativacao    [0, 1]   baixa ↔ alta ativação
    carga       [0, 1]   leve ↔ carregado existencialmente
"""
import numpy as np
from dataclasses import dataclass
from ..token import Token


@dataclass
class EstruturaEmocional:
    valencia: float       # [-1, 1]
    intensidade: float    # [0, 1]
    coerencia: float      # [0, 1]
    ativacao: float       # [0, 1]
    carga: float          # [0, 1]

    @property
    def vetor(self) -> np.ndarray:
        return np.array([
            (self.valencia + 1) / 2,  # normalizado para [0,1]
            self.intensidade,
            self.coerencia,
            self.ativacao,
            self.carga,
        ])

    def __repr__(self):
        return (
            f"EE(valencia={self.valencia:+.3f} "
            f"intensidade={self.intensidade:.3f} "
            f"coerencia={self.coerencia:.3f} "
            f"ativacao={self.ativacao:.3f} "
            f"carga={self.carga:.3f})"
        )


def calcular_estrutura_emocional(tokens: list[Token]) -> EstruturaEmocional:
    """Extrai o vetor emocional dos tokens processados.

    Cada âmbito é derivado diretamente das métricas de frequência:
        valencia    ← balanço entre faixas baixas (calmo) e altas (tenso)
        intensidade ← energia total do sinal normalizada
        coerencia   ← estabilidade harmônica média
        ativacao    ← fração de tokens acima do threshold
        carga       ← concentração de sinal nos tokens mais ativados
    """
    if not tokens:
        return EstruturaEmocional(0.0, 0.0, 0.0, 0.0, 0.0)

    sinal_total = sum(t.sinal_recebido for t in tokens)
    n = len(tokens)

    # Valencia: freq baixa = calmo (+), freq alta = tenso (-)
    valencia = 0.0
    if sinal_total > 0:
        for t in tokens:
            peso = t.sinal_recebido / sinal_total
            f = t.frequencia_dominante
            if f < 0.35:
                valencia += peso * 0.6
            elif f > 0.65:
                valencia -= peso * 0.4
    valencia = float(np.clip(valencia, -1.0, 1.0))

    # Intensidade: energia total / máximo teórico
    intensidade = float(np.clip(sinal_total / (n * 5.0), 0.0, 1.0))

    # Coerência: estabilidade harmônica média
    coerencia = float(np.mean([t.estabilidade for t in tokens]))

    # Ativação: fração de tokens com sinal > 0
    ativacao = float(sum(1 for t in tokens if t.sinal_recebido > 0) / n)

    # Carga existencial: concentração — top 20% dos tokens com quanto do sinal?
    sinais_ord = sorted([t.sinal_recebido for t in tokens], reverse=True)
    top_k = max(1, int(n * 0.2))
    carga = float(np.clip(sum(sinais_ord[:top_k]) / (sinal_total + 1e-10), 0.0, 1.0))

    return EstruturaEmocional(valencia, intensidade, coerencia, ativacao, carga)
