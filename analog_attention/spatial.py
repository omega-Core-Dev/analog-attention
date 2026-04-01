import math
from .token import Token


class GridIndex:
    """Índice espacial baseado em grid uniforme para busca de vizinhos em O(1) amortizado.

    Divide o espaço em células de tamanho `cell_size`. Busca de vizinhos dentro
    de um raio r requer verificar apenas as células adjacentes — O(r²/cell²) células,
    independente do total de tokens n.

    Uso típico:
        idx = GridIndex(tokens, cell_size=raio)
        vizinhos = idx.vizinhos_no_raio(token_origem, raio)
    """

    def __init__(self, tokens: list[Token], cell_size: float):
        if cell_size <= 0:
            raise ValueError("cell_size deve ser > 0")
        self._cell_size = cell_size
        self._grid: dict[tuple[int, int], list[Token]] = {}
        for token in tokens:
            self._inserir(token)

    def _celula(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self._cell_size)), int(math.floor(y / self._cell_size)))

    def _inserir(self, token: Token) -> None:
        cx, cy = self._celula(*token.coordenada)
        self._grid.setdefault((cx, cy), []).append(token)

    def vizinhos_no_raio(self, origem: Token, raio: float) -> list[Token]:
        """Retorna todos os tokens (exceto origem) dentro do raio.

        Complexidade: O(r²/cell²) células verificadas + O(vizinhos) filtro fino.
        Para cell_size ≈ raio, isso é O(9) células = O(1) amortizado.
        """
        ox, oy = origem.coordenada
        span = math.ceil(raio / self._cell_size)
        cx0, cy0 = self._celula(ox, oy)
        resultado = []
        raio2 = raio * raio
        for dcx in range(-span, span + 1):
            for dcy in range(-span, span + 1):
                celula = (cx0 + dcx, cy0 + dcy)
                for token in self._grid.get(celula, []):
                    if token is origem:
                        continue
                    dx = token.coordenada[0] - ox
                    dy = token.coordenada[1] - oy
                    if dx * dx + dy * dy <= raio2:
                        resultado.append(token)
        return resultado
