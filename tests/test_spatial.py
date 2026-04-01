import pytest
from analog_attention.spatial import GridIndex
from analog_attention import Token


def make_tokens():
    return [
        Token("origem", (0.0, 0.0)),
        Token("perto1", (1.0, 0.0)),
        Token("perto2", (0.0, 1.0)),
        Token("longe",  (10.0, 10.0)),
    ]


def test_vizinhos_no_raio_encontra_proximos():
    tokens = make_tokens()
    idx = GridIndex(tokens, cell_size=2.0)
    vizinhos = idx.vizinhos_no_raio(tokens[0], raio=2.0)
    nomes = {t.nome for t in vizinhos}
    assert "perto1" in nomes
    assert "perto2" in nomes


def test_vizinhos_no_raio_exclui_distantes():
    tokens = make_tokens()
    idx = GridIndex(tokens, cell_size=2.0)
    vizinhos = idx.vizinhos_no_raio(tokens[0], raio=2.0)
    nomes = {t.nome for t in vizinhos}
    assert "longe" not in nomes


def test_vizinhos_nao_inclui_origem():
    tokens = make_tokens()
    idx = GridIndex(tokens, cell_size=2.0)
    vizinhos = idx.vizinhos_no_raio(tokens[0], raio=2.0)
    assert tokens[0] not in vizinhos


def test_cell_size_invalido():
    with pytest.raises(ValueError):
        GridIndex([], cell_size=0.0)


def test_grid_vazio():
    idx = GridIndex([], cell_size=1.0)
    t = Token("x", (0.0, 0.0))
    assert idx.vizinhos_no_raio(t, raio=1.0) == []


def test_resultado_consistente_com_forca_bruta():
    import math
    tokens = [Token(str(i), (float(i % 5), float(i // 5))) for i in range(25)]
    raio = 1.5
    idx = GridIndex(tokens, cell_size=raio)
    origem = tokens[12]
    vizinhos_idx = set(t.nome for t in idx.vizinhos_no_raio(origem, raio))
    ox, oy = origem.coordenada
    vizinhos_bruta = {
        t.nome for t in tokens
        if t is not origem and math.hypot(t.coordenada[0] - ox, t.coordenada[1] - oy) <= raio
    }
    assert vizinhos_idx == vizinhos_bruta
