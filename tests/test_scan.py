from analog_attention import Token, scan


def make_tokens():
    return [
        Token("t0", (0.0, 0.0)),
        Token("t1", (1.0, 2.0)),
        Token("t2", (-1.0, 3.0)),
    ]


def test_scan_retorna_mapa():
    tokens = make_tokens()
    mapa = scan(tokens)
    assert mapa == {
        "t0": (0.0, 0.0),
        "t1": (1.0, 2.0),
        "t2": (-1.0, 3.0),
    }


def test_scan_lista_vazia():
    assert scan([]) == {}


def test_scan_nao_modifica_tokens():
    tokens = make_tokens()
    scan(tokens)
    for t in tokens:
        assert t.sinal_recebido == 0.0
