import pytest
from analog_attention.nec import (
    processar_nec, CicloNEC,
    EstruturaEmocional, EstadoAtual, ReferenciaAssociativa,
)


def test_ciclo_basico():
    ciclo = processar_nec("quero entender como funciona", verbose=False)
    assert isinstance(ciclo, CicloNEC)
    assert isinstance(ciclo.resposta.texto, str)
    assert len(ciclo.resposta.texto) > 0


def test_modo_valido():
    ciclo = processar_nec("frequência onda sinal campo", verbose=False)
    modos_validos = {"exploratório", "confirmatório", "adaptativo", "tenso", "estável"}
    assert ciclo.resposta.modo in modos_validos


def test_dois_ciclos_com_ancora():
    c1 = processar_nec("primeiro contexto aqui", verbose=False)
    c2 = processar_nec("segundo contexto diferente", ancora=c1.ancora, verbose=False)
    assert c2.ea.modo in ("contínuo", "salto")
    assert c2.ancora is not None


def test_salto_detectado():
    c1 = processar_nec("calma paz tranquilidade", verbose=False)
    # Input muito diferente deve gerar salto eventualmente
    c2 = processar_nec("urgente critico alerta perigo", ancora=c1.ancora, verbose=False)
    assert isinstance(c2.ea.flag_salto, bool)


def test_estrutura_emocional_ranges():
    ciclo = processar_nec("teste de ranges", verbose=False)
    ee = ciclo.ee
    assert -1.0 <= ee.valencia <= 1.0
    assert 0.0 <= ee.intensidade <= 1.0
    assert 0.0 <= ee.coerencia <= 1.0
    assert 0.0 <= ee.ativacao <= 1.0
    assert 0.0 <= ee.carga <= 1.0


def test_primeiro_ciclo_sem_ancora():
    ciclo = processar_nec("input inicial", verbose=False)
    assert ciclo.ea.modo == "primeiro_ciclo"
    assert ciclo.ra.delta_novidade == 1.0


def test_tokens_chave_presentes():
    ciclo = processar_nec("professor frequência onda", verbose=False)
    assert len(ciclo.resposta.tokens_chave) > 0


def test_ancora_gerada():
    ciclo = processar_nec("gerar ancora", verbose=False)
    assert ciclo.ancora is not None
    assert ciclo.ancora.tokens_ativados >= 0


def test_input_vazio_levanta_erro():
    with pytest.raises(ValueError):
        processar_nec("", verbose=False)


def test_campo_p_normalizado():
    ciclo = processar_nec("campo p normalizado", verbose=False)
    total = ciclo.pp.campo_p.sum()
    assert abs(total - 1.0) < 1e-6
