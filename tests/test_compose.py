import pytest
from analog_attention import Token, compose


def test_compose_cluster_vazio():
    tokens = [Token("a", (0.0, 0.0)), Token("b", (1.0, 0.0))]
    resultado = compose(tokens)
    assert resultado["cluster"] == []
    assert resultado["sinal_total"] == 0.0
    assert resultado["pesos"] == {}


def test_compose_normaliza_pesos():
    t1, t2 = Token("a", (0.0, 0.0)), Token("b", (1.0, 0.0))
    t1.receber_sinal(3.0)
    t2.receber_sinal(1.0)
    resultado = compose([t1, t2])
    assert resultado["pesos"]["a"] == pytest.approx(0.75)
    assert resultado["pesos"]["b"] == pytest.approx(0.25)


def test_compose_sinal_total():
    t1, t2 = Token("a", (0.0, 0.0)), Token("b", (1.0, 0.0))
    t1.receber_sinal(2.0)
    t2.receber_sinal(3.0)
    resultado = compose([t1, t2])
    assert resultado["sinal_total"] == pytest.approx(5.0)


def test_compose_exclui_nao_atingidos():
    t1 = Token("atingido", (0.0, 0.0))
    t2 = Token("nao_atingido", (1.0, 0.0))
    t1.receber_sinal(1.0)
    resultado = compose([t1, t2])
    assert len(resultado["cluster"]) == 1
    assert resultado["cluster"][0].nome == "atingido"
