import pytest
from analog_attention import Token, propagate


def make_sala():
    return [
        Token("frente_esq", (1.0, 1.0)),
        Token("frente_dir", (-1.0, 1.0)),
        Token("meio",       (0.0, 3.0)),
        Token("fundo",      (0.0, 8.0)),
    ]


def test_propagate_retorna_tokens():
    tokens = make_sala()
    resultado = propagate(tokens)
    assert resultado is tokens


def test_tokens_perto_recebem_mais_sinal():
    tokens = make_sala()
    propagate(tokens, intensidade_professor=5.0)
    sinal_frente = tokens[0].sinal_recebido
    sinal_fundo = tokens[3].sinal_recebido
    assert sinal_frente > sinal_fundo


def test_todos_tokens_atingidos_com_alta_intensidade():
    tokens = make_sala()
    propagate(tokens, intensidade_professor=50.0)
    for t in tokens:
        assert t.foi_atingido


def test_raio_limita_propagacao_social():
    tokens_com_raio = [
        Token("origem", (0.0, 0.0)),
        Token("perto",  (1.0, 0.0)),
        Token("longe",  (20.0, 0.0)),
    ]
    tokens_sem_raio = [
        Token("origem", (0.0, 0.0)),
        Token("perto",  (1.0, 0.0)),
        Token("longe",  (20.0, 0.0)),
    ]
    propagate(tokens_com_raio, intensidade_professor=10.0, raio=3.0, threshold=0.0)
    propagate(tokens_sem_raio, intensidade_professor=10.0, raio=None, threshold=0.0)

    # Com raio, o token longe deve receber menos sinal social do que sem raio
    assert tokens_com_raio[2].sinal_recebido <= tokens_sem_raio[2].sinal_recebido


def test_falloff_quadratic():
    tokens = make_sala()
    propagate(tokens, intensidade_professor=5.0, falloff="quadratic")
    assert all(t.sinal_recebido >= 0 for t in tokens)


def test_falloff_linear():
    tokens = make_sala()
    propagate(tokens, intensidade_professor=5.0, falloff="linear")
    assert all(t.sinal_recebido >= 0 for t in tokens)
