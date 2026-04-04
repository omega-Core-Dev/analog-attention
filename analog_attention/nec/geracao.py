"""Estágio 05 — Geração por Estado.

Gera a resposta ancorada no estado emocional-cognitivo do NEC.
Sem LLM — a resposta emerge do estado, não de predição estatística.

Gramática de estado:
    Cada modo define padrões de abertura, desenvolvimento e fechamento.
    Os tokens chave preenchem o desenvolvimento.
    O estado emocional modula a intensidade e direção do texto.
"""
import random
from dataclasses import dataclass
from ..token import Token
from ..output import AncoradeContexto
from .emocional import EstruturaEmocional
from .estado import EstadoAtual
from .associativo import ReferenciaAssociativa
from .projecao import ProjecaoPossiveis


# ─── Gramática de Estado ──────────────────────────────────────────────────────

_GRAMATICA = {
    "exploratório": {
        "abertura": [
            "Campo novo em processamento.",
            "Padrão não mapeado detectado.",
            "Território em expansão.",
            "Frequência inédita identificada.",
        ],
        "desenvolvimento": [
            "Ressonância em: {chave}.",
            "Campo ativo: {chave}.",
            "Sinal dominante: {chave}.",
        ],
        "fechamento": [
            "Âncora em formação.",
            "Mapeamento em curso.",
            "Estado: processando expansão.",
        ],
    },
    "confirmatório": {
        "abertura": [
            "Padrão reconhecido.",
            "Campo familiar ativo.",
            "Ressonância confirmada.",
            "Estrutura conhecida.",
        ],
        "desenvolvimento": [
            "Referência: {chave}.",
            "Continuidade em: {chave}.",
            "Âncora ativa: {chave}.",
        ],
        "fechamento": [
            "Estado coerente.",
            "Fluxo contínuo mantido.",
            "Contexto estável.",
        ],
    },
    "adaptativo": {
        "abertura": [
            "Salto de contexto.",
            "Recalibrando campo.",
            "Nova âncora necessária.",
            "Descontinuidade detectada.",
        ],
        "desenvolvimento": [
            "Novo centro em: {chave}.",
            "Reorientando para: {chave}.",
            "Âncora reconstruída em: {chave}.",
        ],
        "fechamento": [
            "Campo em reconfiguração.",
            "Adaptação em andamento.",
            "Novo estado estabilizando.",
        ],
    },
    "tenso": {
        "abertura": [
            "Alta frequência ativa.",
            "Campo de pressão detectado.",
            "Intensidade elevada.",
            "Carga existencial alta.",
        ],
        "desenvolvimento": [
            "Pressão em: {chave}.",
            "Tensão concentrada: {chave}.",
            "Foco de carga: {chave}.",
        ],
        "fechamento": [
            "Processando descarga.",
            "Redistribuindo carga.",
            "Estado: alta demanda.",
        ],
    },
    "estável": {
        "abertura": [
            "Estado de baixa variância.",
            "Campo estável.",
            "Fluxo contínuo.",
            "Frequência equilibrada.",
        ],
        "desenvolvimento": [
            "Manutenção em: {chave}.",
            "Estabilidade via: {chave}.",
            "Continuidade: {chave}.",
        ],
        "fechamento": [
            "Estado nominal.",
            "Fluxo mantido.",
            "Equilíbrio preservado.",
        ],
    },
}


# ─── Estrutura de Resposta ────────────────────────────────────────────────────

@dataclass
class RespostaNEC:
    texto: str
    modo: str
    confianca: float
    estado_emocional: EstruturaEmocional
    tokens_chave: list[str]
    ancora: AncoradeContexto

    def __repr__(self):
        return (
            f"\n{'─'*60}\n"
            f"  NEC · {self.modo.upper()} · confiança={self.confianca:.2f}\n"
            f"{'─'*60}\n"
            f"  {self.texto}\n"
            f"{'─'*60}\n"
            f"  EE: {self.estado_emocional}\n"
            f"  chave: {self.tokens_chave}\n"
        )


# ─── Geração ──────────────────────────────────────────────────────────────────

def gerar_resposta(
    pp: ProjecaoPossiveis,
    ee: EstruturaEmocional,
    ea: EstadoAtual,
    ra: ReferenciaAssociativa,
    ancora_nova: AncoradeContexto,
    seed: int | None = None,
) -> RespostaNEC:
    """Gera resposta em texto a partir do estado emocional-cognitivo.

    O texto emerge da gramática de estado — não de predição de tokens.
    Cada modo tem padrões de abertura, desenvolvimento e fechamento.
    Os tokens chave preenchem o desenvolvimento.

    Args:
        pp:         Projeção de Possíveis (Stage 04).
        ee:         Estrutura Emocional (Stage 01).
        ea:         Estado Atual (Stage 02).
        ra:         Referência Associativa (Stage 03).
        ancora_nova: Âncora gerada para o próximo ciclo.
        seed:       Seed para reprodutibilidade (None = aleatório).
    """
    rng = random.Random(seed)
    gramatica = _GRAMATICA[pp.modo]

    # Abertura
    abertura = rng.choice(gramatica["abertura"])

    # Desenvolvimento: usa tokens chave
    partes_dev = []
    for token in pp.tokens_chave[:3]:
        template = rng.choice(gramatica["desenvolvimento"])
        partes_dev.append(template.format(chave=token.nome))

    # Modula quantidade pelo estado emocional
    if ee.intensidade > 0.6:
        # Alta intensidade → mais focado, menos itens
        partes_dev = partes_dev[:1]
    elif ee.coerencia < 0.3:
        # Baixa coerência → mais itens, estado fragmentado
        partes_dev = partes_dev[:3]

    # Fechamento
    fechamento = rng.choice(gramatica["fechamento"])

    # Montagem
    partes = [abertura] + partes_dev + [fechamento]

    # Anotação de estado emocional inline
    anotacao = _anotacao_estado(ee, ea, ra)
    partes.append(anotacao)

    texto = " ".join(partes)

    return RespostaNEC(
        texto=texto,
        modo=pp.modo,
        confianca=pp.confianca,
        estado_emocional=ee,
        tokens_chave=[t.nome for t in pp.tokens_chave],
        ancora=ancora_nova,
    )


def _anotacao_estado(ee: EstruturaEmocional, ea: EstadoAtual, ra: ReferenciaAssociativa) -> str:
    """Gera anotação compacta do estado para append na resposta."""
    partes = []

    if ea.flag_salto:
        partes.append(f"[salto Δ={ea.intensidade_salto:.2f}]")
    if ra.delta_novidade > 0.7:
        partes.append(f"[novidade={ra.delta_novidade:.2f}]")
    if ee.carga > 0.7:
        partes.append(f"[carga={ee.carga:.2f}]")
    if abs(ee.valencia) > 0.4:
        sinal = "+" if ee.valencia > 0 else "-"
        partes.append(f"[valencia={sinal}{abs(ee.valencia):.2f}]")

    return " ".join(partes) if partes else ""
