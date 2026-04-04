"""NEC — Núcleo Emocional-Cognitivo.

Arquitetura baseada em experiência direta humana (QE-05).
Processa texto em tempo real através de 5 estágios emocionais-cognitivos.
Sem LLM — a resposta emerge do estado interno.

Uso:
    from analog_attention.nec import processar_nec

    ciclo = processar_nec("seu texto aqui")
    ciclo2 = processar_nec("continuação", ancora=ciclo.ancora)
"""
from .emocional import EstruturaEmocional, calcular_estrutura_emocional
from .estado import EstadoAtual, calcular_estado_atual
from .associativo import ReferenciaAssociativa, calcular_referencia_associativa
from .operador import operador_theta
from .projecao import ProjecaoPossiveis, calcular_projecao
from .geracao import RespostaNEC, gerar_resposta
from .nucleo import CicloNEC, processar_nec

__all__ = [
    "processar_nec",
    "CicloNEC",
    "EstruturaEmocional",
    "EstadoAtual",
    "ReferenciaAssociativa",
    "ProjecaoPossiveis",
    "RespostaNEC",
    "calcular_estrutura_emocional",
    "calcular_estado_atual",
    "calcular_referencia_associativa",
    "operador_theta",
    "calcular_projecao",
    "gerar_resposta",
]
