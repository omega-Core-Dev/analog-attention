import numpy as np
from analog_attention.tokenizer import tokenizar, assinatura_do_texto
from analog_attention.token import Token


def test_tokenizar_retorna_tokens():
    tokens = tokenizar("olá mundo")
    assert len(tokens) == 2
    assert all(isinstance(t, Token) for t in tokens)


def test_tokenizar_vazio():
    assert tokenizar("") == []
    assert tokenizar("   ") == []


def test_token_tem_assinatura():
    tokens = tokenizar("teste")
    assert tokens[0].assinatura.sum() > 0


def test_token_tem_classe():
    tokens = tokenizar("frequência")
    assert tokens[0].classe.startswith("ONDA_")


def test_pertencimento_entre_tokens():
    tokens = tokenizar("a b c")
    assert "b" in tokens[0].pertence_a
    assert "c" in tokens[0].pertence_a


def test_assinatura_do_texto():
    mag = assinatura_do_texto("entrada do professor")
    assert len(mag) == 16
    assert abs(mag.sum() - 1.0) < 1e-6


def test_tokens_distintos():
    tokens = tokenizar("quero entender")
    assert not np.allclose(tokens[0].assinatura, tokens[1].assinatura)
