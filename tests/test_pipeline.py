import pytest
from analog_attention import processar
from analog_attention.output import FrequencyLog, AncoradeContexto


def test_pipeline_basico():
    log, ancora = processar("teste de entrada", verbose=False)
    assert isinstance(log, FrequencyLog)
    assert isinstance(ancora, AncoradeContexto)


def test_pipeline_dois_ciclos():
    log1, ancora1 = processar("primeiro ciclo", verbose=False)
    log2, ancora2 = processar("segundo ciclo", ancara=ancora1, verbose=False)
    assert log2.f_peak != log1.f_peak or log2.sinal_total != log1.sinal_total


def test_ancora_tem_assinatura():
    _, ancora = processar("input qualquer", verbose=False)
    assert ancora.assinatura.sum() >= 0
    assert ancora.tokens_ativados > 0


def test_log_tem_registros():
    log, _ = processar("frequência onda sinal", verbose=False)
    assert len(log.registros) == 3


def test_pipeline_vazio():
    with pytest.raises(ValueError):
        processar("", verbose=False)


def test_tokens_passaram_range():
    log, _ = processar("a b c d e", verbose=False)
    assert 0 <= log.tokens_passaram <= len(log.registros)
