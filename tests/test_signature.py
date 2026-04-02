import numpy as np
import pytest
from analog_attention.signature import compute_signature, frequencia_dominante, estabilidade, correlacao


def test_signature_shape():
    mag, fase = compute_signature("hello")
    assert len(mag) == 16
    assert len(fase) == 16


def test_signature_normalizada():
    mag, _ = compute_signature("teste")
    assert abs(mag.sum() - 1.0) < 1e-6


def test_signature_vazia():
    mag, fase = compute_signature("")
    assert mag.sum() == 0.0


def test_tokens_diferentes_tem_assinaturas_diferentes():
    mag_a, _ = compute_signature("quero")
    mag_b, _ = compute_signature("sistema")
    assert not np.allclose(mag_a, mag_b)


def test_mesma_string_assinatura_deterministica():
    mag1, _ = compute_signature("frequência")
    mag2, _ = compute_signature("frequência")
    assert np.allclose(mag1, mag2)


def test_frequencia_dominante_range():
    mag, _ = compute_signature("onda")
    f = frequencia_dominante(mag)
    assert 0.0 <= f <= 1.0


def test_estabilidade_range():
    mag, _ = compute_signature("token")
    e = estabilidade(mag)
    assert 0.0 <= e <= 1.0


def test_correlacao_identica():
    mag, _ = compute_signature("mesmo")
    assert correlacao(mag, mag) == pytest.approx(1.0)


def test_correlacao_range():
    mag_a, _ = compute_signature("alpha")
    mag_b, _ = compute_signature("beta")
    c = correlacao(mag_a, mag_b)
    assert 0.0 <= c <= 1.0


def test_correlacao_zeros():
    assert correlacao(np.zeros(8), np.zeros(8)) == 0.0
