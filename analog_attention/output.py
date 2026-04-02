"""Log de saída em frequências com limiares parabólicos.

Saída estruturada para análise da assinatura de ativação —
base para encaixar geração por limiares parabólicos.
"""
import numpy as np
from dataclasses import dataclass, field
from .token import Token
from .signature import frequencia_dominante


# ─── Limiar parabólico ────────────────────────────────────────────────────────

def limiar_parabolico(
    f: float,
    f_peak: float,
    base: float = 0.1,
    amplitude: float = 0.9,
    curvatura: float = 4.0,
) -> float:
    """Limiar de ativação em função da frequência do token.

    threshold(f) = base + amplitude · max(0, 1 − κ · (f − f_peak)²)

    Tokens na frequência do pico têm o maior limiar (exigência maior).
    Tokens distantes do pico têm limiar menor (entram pela banda lateral).
    """
    val = max(0.0, 1.0 - curvatura * (f - f_peak) ** 2)
    return base + amplitude * val


# ─── Âncora de contexto (dois ciclos) ─────────────────────────────────────────

@dataclass
class AncoradeContexto:
    """Pegada do ciclo anterior — viés de frequência para o próximo ciclo."""
    assinatura: np.ndarray
    f_peak: float
    intensidade: float
    tokens_ativados: int

    def blend(self, query: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Mistura a âncora com a nova query. alpha = peso da âncora."""
        resultado = (1.0 - alpha) * query + alpha * self.assinatura
        total = resultado.sum()
        return resultado / total if total > 0 else resultado


# ─── Log de frequências ───────────────────────────────────────────────────────

@dataclass
class RegistroToken:
    nome: str
    classe: str
    subclasse: str
    frequencia: float
    sinal: float
    limiar: float
    passa: bool
    estabilidade: float


@dataclass
class FrequencyLog:
    """Saída completa do ciclo de atenção em frequências."""
    f_peak: float
    assinatura_cluster: list[float]
    registros: list[RegistroToken]
    tokens_passaram: int
    sinal_total: float
    ancara: AncoradeContexto


def gerar_log(
    tokens: list[Token],
    query: np.ndarray,
    curvatura_limiar: float = 4.0,
    base_limiar: float = 0.1,
) -> FrequencyLog:
    """Gera o log de frequências após um ciclo de atenção.

    Args:
        tokens:           tokens após propagação.
        query:            assinatura do professor (input completo).
        curvatura_limiar: curvatura da parábola de limiar.
        base_limiar:      limiar mínimo base.

    Returns:
        FrequencyLog com análise completa + âncora para próximo ciclo.
    """
    f_peak = frequencia_dominante(query)
    ativados = [t for t in tokens if t.foi_atingido]

    # Assinatura agregada do cluster (média ponderada pelo sinal)
    if ativados:
        pesos = np.array([t.sinal_recebido for t in ativados])
        pesos = pesos / pesos.sum()
        assinatura_cluster = sum(p * t.assinatura for p, t in zip(pesos, ativados))
    else:
        assinatura_cluster = np.zeros_like(query)

    sinal_total = sum(t.sinal_recebido for t in ativados)

    # Registros por token
    registros = []
    for t in tokens:
        f = t.frequencia_dominante
        limiar = limiar_parabolico(f, f_peak, base=base_limiar, curvatura=curvatura_limiar)
        registros.append(RegistroToken(
            nome=t.nome,
            classe=t.classe,
            subclasse=t.subclasse,
            frequencia=round(f, 4),
            sinal=round(t.sinal_recebido, 4),
            limiar=round(limiar, 4),
            passa=t.sinal_recebido >= limiar,
            estabilidade=round(t.estabilidade, 4),
        ))

    # Ordena por sinal decrescente
    registros.sort(key=lambda r: r.sinal, reverse=True)

    ancara = AncoradeContexto(
        assinatura=assinatura_cluster,
        f_peak=float(frequencia_dominante(assinatura_cluster)) if ativados else f_peak,
        intensidade=sinal_total,
        tokens_ativados=len(ativados),
    )

    return FrequencyLog(
        f_peak=round(f_peak, 4),
        assinatura_cluster=assinatura_cluster.tolist(),
        registros=registros,
        tokens_passaram=sum(1 for r in registros if r.passa),
        sinal_total=round(sinal_total, 4),
        ancara=ancara,
    )


def imprimir_log(log: FrequencyLog) -> None:
    """Imprime o log de frequências formatado para análise."""
    print(f"\n{'═' * 72}")
    print(f"  FREQUENCY LOG  |  f_peak={log.f_peak:.4f}  |  sinal_total={log.sinal_total:.4f}")
    print(f"  tokens_passaram={log.tokens_passaram}/{len(log.registros)}")
    print(f"{'═' * 72}")
    print(f"  {'TOKEN':<14} {'CLASSE':<18} {'FREQ':>6}  {'SINAL':>7}  {'LIMIAR':>7}  {'ESTAB':>5}  {''}")
    print(f"  {'-' * 68}")
    for r in log.registros:
        marca = "✓" if r.passa else "·"
        print(
            f"  {r.nome:<14} {r.classe:<18} {r.frequencia:>6.4f}  "
            f"{r.sinal:>7.4f}  {r.limiar:>7.4f}  {r.estabilidade:>5.3f}  {marca}"
        )
    print(f"{'═' * 72}")
    print(f"  âncora → f_peak={log.ancara.f_peak:.4f}  tokens_ativados={log.ancara.tokens_ativados}\n")
