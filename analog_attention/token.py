class Token:
    def __init__(self, nome: str, coordenada: tuple[float, float]):
        self.nome = nome
        self.coordenada = coordenada
        self.sinal_recebido: float = 0.0

    def receber_sinal(self, sinal: float) -> None:
        self.sinal_recebido += sinal

    @property
    def foi_atingido(self) -> bool:
        return self.sinal_recebido > 0

    def __repr__(self) -> str:
        return f"Token({self.nome!r}, coord={self.coordenada}, sinal={self.sinal_recebido:.4f})"
