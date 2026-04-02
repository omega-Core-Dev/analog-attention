import numpy as np
from dataclasses import dataclass, field


@dataclass
class Token:
    """Unidade atômica do fluxo — carrega identidade de frequência, não apenas texto.

    Campos imutáveis (definidos na tokenização):
        nome, assinatura, fase, classe, subclasse,
        magnitude_total, estabilidade, pertence_a

    Campo de estado (atualizado durante atenção):
        sinal_recebido
    """
    nome: str
    assinatura: np.ndarray       # magnitude FFT normalizada [N_COMPONENTS]
    fase: np.ndarray             # fase FFT [N_COMPONENTS]
    classe: str                  # classe de onda (BASA)
    subclasse: str               # subclasse de onda (BASA)
    magnitude_total: float       # energia total do sinal
    estabilidade: float          # estabilidade harmônica [0, 1]
    pertence_a: list[str] = field(default_factory=list)  # nomes dos co-tokens
    sinal_recebido: float = 0.0  # sinal acumulado de atenção

    @property
    def frequencia_dominante(self) -> float:
        """Frequência dominante normalizada [0, 1]."""
        return float(np.argmax(self.assinatura)) / len(self.assinatura)

    @property
    def foi_atingido(self) -> bool:
        return self.sinal_recebido > 0.0

    def receber_sinal(self, sinal: float) -> None:
        self.sinal_recebido += sinal

    def reset(self) -> None:
        self.sinal_recebido = 0.0

    def __repr__(self) -> str:
        return (
            f"Token({self.nome!r}, "
            f"classe={self.classe}, "
            f"freq={self.frequencia_dominante:.3f}, "
            f"estab={self.estabilidade:.3f}, "
            f"sinal={self.sinal_recebido:.4f})"
        )

    def __hash__(self):
        return hash(self.nome)

    def __eq__(self, other):
        return isinstance(other, Token) and self.nome == other.nome
