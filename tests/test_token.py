import pytest
from analog_attention import Token


def test_initial_state():
    t = Token("a", (1.0, 2.0))
    assert t.nome == "a"
    assert t.coordenada == (1.0, 2.0)
    assert t.sinal_recebido == 0.0
    assert t.foi_atingido is False


def test_receber_sinal_acumula():
    t = Token("b", (0.0, 0.0))
    t.receber_sinal(1.0)
    t.receber_sinal(0.5)
    assert t.sinal_recebido == pytest.approx(1.5)


def test_foi_atingido_apos_sinal():
    t = Token("c", (1.0, 1.0))
    t.receber_sinal(0.01)
    assert t.foi_atingido is True


def test_sinal_zero_nao_atingido():
    t = Token("d", (1.0, 1.0))
    assert t.foi_atingido is False


def test_repr():
    t = Token("e", (1.0, 0.0))
    assert "e" in repr(t)


def test_reset_zera_sinal():
    t = Token("f", (1.0, 0.0))
    t.receber_sinal(5.0)
    t.reset()
    assert t.sinal_recebido == 0.0
    assert t.foi_atingido is False


def test_reset_permite_reusar():
    t = Token("g", (1.0, 0.0))
    t.receber_sinal(3.0)
    t.reset()
    t.receber_sinal(1.0)
    assert t.sinal_recebido == pytest.approx(1.0)
