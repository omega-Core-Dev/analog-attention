import pytest
from analog_attention import Token, candidate


def test_candidate_acima_do_threshold():
    t = Token("a", (0.0, 0.0))
    t.receber_sinal(1.0)
    assert candidate(t, threshold=0.4) is True


def test_candidate_abaixo_do_threshold():
    t = Token("b", (0.0, 0.0))
    t.receber_sinal(0.1)
    assert candidate(t, threshold=0.4) is False


def test_candidate_exatamente_no_threshold():
    t = Token("c", (0.0, 0.0))
    t.receber_sinal(0.4)
    assert candidate(t, threshold=0.4) is True


def test_candidate_sem_sinal():
    t = Token("d", (0.0, 0.0))
    assert candidate(t) is False


def test_candidate_threshold_zero():
    t = Token("e", (0.0, 0.0))
    assert candidate(t, threshold=0.0) is True
