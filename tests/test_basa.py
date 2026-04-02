from analog_attention.basa import classificar, faixa_da_classe, WAVE_CLASSES


def test_todas_frequencias_tem_classe():
    for f in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        classe, subclasse = classificar(f)
        assert isinstance(classe, str) and len(classe) > 0
        assert isinstance(subclasse, str) and len(subclasse) > 0


def test_faixas_nao_se_sobrepoem():
    faixas = [(wc.faixa_min, wc.faixa_max) for wc in WAVE_CLASSES]
    for i in range(len(faixas) - 1):
        assert faixas[i][1] == faixas[i + 1][0]


def test_faixa_da_classe_valida():
    resultado = faixa_da_classe("ONDA_MEDIA")
    assert resultado is not None
    fmin, fmax = resultado
    assert fmin < fmax


def test_faixa_da_classe_invalida():
    assert faixa_da_classe("CLASSE_INEXISTENTE") is None
