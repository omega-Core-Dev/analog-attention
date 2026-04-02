"""Pipeline completo: texto humano → log de frequências.

Dois ciclos com âncora de contexto:
    Ciclo N   → processa input, gera log + âncora
    Ciclo N+1 → query blendada com âncora do ciclo anterior
"""
import numpy as np
from .tokenizer import tokenizar, assinatura_do_texto
from .propagate import propagate
from .output import gerar_log, imprimir_log, AncoradeContexto, FrequencyLog


def processar(
    texto: str,
    ancara: AncoradeContexto | None = None,
    intensidade: float = 5.0,
    curvatura_broadcast: float = 4.0,
    curvatura_limiar: float = 4.0,
    fator_social: float = 0.5,
    threshold: float = 0.3,
    max_hops: int = 3,
    alpha_ancora: float = 0.3,
    verbose: bool = True,
) -> tuple[FrequencyLog, AncoradeContexto]:
    """Processa um ciclo completo de input humano.

    Args:
        texto:               input humano em texto.
        ancara:              âncora do ciclo anterior (None = primeiro ciclo).
        intensidade:         amplitude do broadcast parabólico.
        curvatura_broadcast: curvatura κ do broadcast.
        curvatura_limiar:    curvatura κ do limiar parabólico no log.
        fator_social:        atenuação por hop na cadeia harmônica.
        threshold:           limiar mínimo para ser candidato.
        max_hops:            hops máximos por candidato.
        alpha_ancora:        peso da âncora no blend da query (0 = sem âncora).
        verbose:             imprime o log.

    Returns:
        (FrequencyLog do ciclo, AncoradeContexto para o próximo ciclo)
    """
    # Tokeniza
    tokens = tokenizar(texto)
    if not tokens:
        raise ValueError("Input vazio após tokenização.")

    # Query do professor = assinatura do texto completo
    query = assinatura_do_texto(texto)

    # Blend com âncora do ciclo anterior (dois ciclos de contexto)
    if ancara is not None:
        query = ancara.blend(query, alpha=alpha_ancora)

    # Propagação
    propagate(
        tokens, query,
        intensidade=intensidade,
        curvatura=curvatura_broadcast,
        fator_social=fator_social,
        threshold=threshold,
        max_hops=max_hops,
    )

    # Log de frequências
    log = gerar_log(tokens, query, curvatura_limiar=curvatura_limiar)

    if verbose:
        imprimir_log(log)

    return log, log.ancara
