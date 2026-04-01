import pytest
from analog_attention import Token, broadcast


def test_broadcast_quadratic_origem():
    # Na origem distância=0 → sinal máximo = intensidade
    t = Token("a", (0.0, 0.0))
    s = broadcast(5.0, t, falloff="quadratic")
    assert s == pytest.approx(5.0)


def test_broadcast_quadratic_distante():
    # Sinal decresce com distância
    perto = Token("p", (1.0, 0.0))
    longe = Token("l", (5.0, 0.0))
    assert broadcast(5.0, perto, falloff="quadratic") > broadcast(5.0, longe, falloff="quadratic")


def test_broadcast_sqrt_falloff():
    t_perto = Token("p", (1.0, 0.0))
    t_longe = Token("l", (9.0, 0.0))
    assert broadcast(5.0, t_perto, falloff="sqrt") > broadcast(5.0, t_longe, falloff="sqrt")


def test_broadcast_linear_falloff():
    t_perto = Token("p", (1.0, 0.0))
    t_longe = Token("l", (9.0, 0.0))
    assert broadcast(5.0, t_perto, falloff="linear") > broadcast(5.0, t_longe, falloff="linear")


def test_broadcast_nunca_negativo():
    t = Token("z", (100.0, 100.0))
    assert broadcast(1.0, t, ruido=999.0, falloff="quadratic") == 0.0


def test_broadcast_falloff_invalido():
    t = Token("a", (1.0, 0.0))
    with pytest.raises(ValueError):
        broadcast(5.0, t, falloff="invalido")


def test_broadcast_nao_modifica_token():
    t = Token("a", (1.0, 0.0))
    broadcast(5.0, t)
    assert t.sinal_recebido == 0.0
