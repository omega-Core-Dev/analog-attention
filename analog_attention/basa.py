"""BASA — Base de Assinaturas de Onda (estática no protótipo).

Classes e subclasses definidas por modulação de frequência / comprimento de onda.
Estrutura modular: substitua WAVE_CLASSES pelo seu schema quando estiver pronto.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ClasseOnda:
    nome: str
    faixa_min: float   # frequência normalizada mínima [0, 1]
    faixa_max: float   # frequência normalizada máxima [0, 1]
    subclasses: tuple[str, ...]
    comprimento: str   # descrição do comprimento de onda


# Registro estático de classes de onda
# MODULAR: o usuário define suas próprias classes neste espaço
WAVE_CLASSES: list[ClasseOnda] = [
    ClasseOnda("ONDA_INFRA",      0.000, 0.100, ("infra_0", "infra_1"),             "extra-longo"),
    ClasseOnda("ONDA_ULTRA_BAIXA",0.100, 0.200, ("ub_0", "ub_1"),                  "muito longo"),
    ClasseOnda("ONDA_BAIXA",      0.200, 0.350, ("baixa_0", "baixa_1", "baixa_2"), "longo"),
    ClasseOnda("ONDA_MEDIA_BAIXA",0.350, 0.500, ("mb_0", "mb_1", "mb_2"),          "medio-longo"),
    ClasseOnda("ONDA_MEDIA",      0.500, 0.650, ("media_0", "media_1", "media_2"), "medio"),
    ClasseOnda("ONDA_MEDIA_ALTA", 0.650, 0.800, ("ma_0", "ma_1", "ma_2"),          "medio-curto"),
    ClasseOnda("ONDA_ALTA",       0.800, 0.920, ("alta_0", "alta_1"),              "curto"),
    ClasseOnda("ONDA_ULTRA_ALTA", 0.920, 1.001, ("ua_0", "ua_1"),                  "ultra-curto"),
]


def classificar(freq: float) -> tuple[str, str]:
    """Retorna (classe, subclasse) para uma frequência dominante normalizada."""
    for wc in WAVE_CLASSES:
        if wc.faixa_min <= freq < wc.faixa_max:
            span = wc.faixa_max - wc.faixa_min
            idx = int((freq - wc.faixa_min) / span * len(wc.subclasses))
            idx = min(idx, len(wc.subclasses) - 1)
            return wc.nome, wc.subclasses[idx]
    return "ONDA_ULTRA_ALTA", "ua_1"


def faixa_da_classe(nome_classe: str) -> tuple[float, float] | None:
    """Retorna (faixa_min, faixa_max) de uma classe por nome."""
    for wc in WAVE_CLASSES:
        if wc.nome == nome_classe:
            return wc.faixa_min, wc.faixa_max
    return None
