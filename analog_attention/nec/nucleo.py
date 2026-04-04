"""NEC — Núcleo Emocional-Cognitivo.

Orquestra os 5 estágios + Operador Θ em tempo real.
Sem LLM — a resposta emerge do estado interno.

Pipeline:
    texto
      ↓ tokenizar (analog_attention)
      ↓ propagate (analog_attention)
      ↓ [01] Estrutura Emocional
      ↓ [02] Estado Atual
      ↓ [03] Referência Associativa
      ↓  Operador Θ → campo P
      ↓ [04] Projeção de Possíveis
      ↓ [05] Geração por Estado
      → RespostaNEC + nova âncora
"""
from dataclasses import dataclass
from ..tokenizer import tokenizar, assinatura_do_texto
from ..propagate import propagate
from ..output import AncoradeContexto, gerar_log
from .emocional import EstruturaEmocional, calcular_estrutura_emocional
from .estado import EstadoAtual, calcular_estado_atual
from .associativo import ReferenciaAssociativa, calcular_referencia_associativa
from .operador import operador_theta
from .projecao import ProjecaoPossiveis, calcular_projecao
from .geracao import RespostaNEC, gerar_resposta


@dataclass
class CicloNEC:
    """Resultado completo de um ciclo do NEC."""
    resposta: RespostaNEC
    ee: EstruturaEmocional
    ea: EstadoAtual
    ra: ReferenciaAssociativa
    pp: ProjecaoPossiveis
    ancora: AncoradeContexto

    def imprimir(self):
        print(self.resposta)


def processar_nec(
    texto: str,
    ancora: AncoradeContexto | None = None,
    intensidade: float = 5.0,
    curvatura: float = 4.0,
    fator_social: float = 0.5,
    threshold: float = 0.3,
    max_hops: int = 3,
    alpha_ancora: float = 0.3,
    top_k_tokens: int = 5,
    verbose: bool = True,
) -> CicloNEC:
    """Executa um ciclo completo do NEC.

    Args:
        texto:          Input humano em texto.
        ancora:         Âncora do ciclo anterior (None = primeiro ciclo).
        intensidade:    Amplitude do broadcast parabólico.
        curvatura:      Curvatura κ do broadcast.
        fator_social:   Atenuação por hop na cadeia harmônica.
        threshold:      Limiar mínimo para candidato.
        max_hops:       Hops máximos por candidato.
        alpha_ancora:   Peso da âncora no blend da query (dois ciclos).
        top_k_tokens:   Tokens chave para geração.
        verbose:        Imprime a resposta.

    Returns:
        CicloNEC com resposta gerada e nova âncora.
    """
    if not texto.strip():
        raise ValueError("Input vazio.")

    # ── Pré-processamento ──────────────────────────────────────────
    tokens = tokenizar(texto)
    query = assinatura_do_texto(texto)

    # Blend com âncora (dois ciclos de contexto)
    if ancora is not None:
        query = ancora.blend(query, alpha=alpha_ancora)

    # Propagação de atenção por frequência
    propagate(
        tokens, query,
        intensidade=intensidade,
        curvatura=curvatura,
        fator_social=fator_social,
        threshold=threshold,
        max_hops=max_hops,
    )

    # Log de frequências (gera nova âncora)
    freq_log = gerar_log(tokens, query)
    ancora_nova = freq_log.ancara

    # ── Estágio 01 ─────────────────────────────────────────────────
    ee = calcular_estrutura_emocional(tokens)

    # ── Estágio 02 ─────────────────────────────────────────────────
    ea = calcular_estado_atual(ee, ancora)

    # ── Estágio 03 ─────────────────────────────────────────────────
    ra = calcular_referencia_associativa(tokens, ea, ancora)

    # ── Operador Θ ─────────────────────────────────────────────────
    campo_p = operador_theta(ee, ea, ra)

    # ── Estágio 04 ─────────────────────────────────────────────────
    pp = calcular_projecao(tokens, ee, ea, ra, campo_p, top_k=top_k_tokens)

    # ── Estágio 05 ─────────────────────────────────────────────────
    resposta = gerar_resposta(pp, ee, ea, ra, ancora_nova)

    ciclo = CicloNEC(
        resposta=resposta,
        ee=ee,
        ea=ea,
        ra=ra,
        pp=pp,
        ancora=ancora_nova,
    )

    if verbose:
        ciclo.imprimir()

    return ciclo
