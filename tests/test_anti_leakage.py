"""
Testes anti-vazamento.

O Apêndice C do livro oferece um checklist de aceite que um humano percorre
e assina. Revisão manual falha por fadiga — que é exatamente o modo de
falha descrito no Capítulo 1. Os itens automatizáveis daquele checklist
viram asserções aqui e passam a rodar a cada commit.

    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from posso_confiar.comparacao import (
    bootstrap_ci, comparacao_bonferroni, gate_f3, teste_permutacao_pareado,
)
from posso_confiar.dados import gerar_dataset
from posso_confiar.databuilder import (
    VazamentoDetectado, auditar_whitelist, construir_features,
    montar_feature_cols, target_encoding_seguro, verifica_defasagem,
)
from posso_confiar.metricas import classificar_regime, skill_score_forecast, wmape
from posso_confiar.validation import verificar_causalidade, walk_forward


@pytest.fixture(scope='module')
def dados():
    return gerar_dataset(n_dias=400, seed=7)


@pytest.fixture(scope='module')
def serie_smooth(dados):
    return dados[dados.sku == 'SKU_00'].reset_index(drop=True)


# ===========================================================================
# Checklist C.1 — whitelist de features
# ===========================================================================

def test_whitelist_rejeita_target():
    df = pd.DataFrame(columns=['data', 'unidades', 'sku', 'lag_7', 'roll_mean_30'])
    cols = montar_feature_cols(df)
    assert 'unidades' not in cols, "o target vazou para as features"
    assert 'data' not in cols
    assert 'sku' not in cols
    assert set(cols) == {'lag_7', 'roll_mean_30'}


def test_whitelist_rejeita_prefixo_nao_declarado():
    """Whitelist é de inclusão: o que não foi declarado não entra."""
    df = pd.DataFrame(columns=['lag_7', 'media_movel_30', 'temp_max'])
    assert montar_feature_cols(df) == ['lag_7']


def test_auditoria_falha_barulhento():
    """Auditoria levanta exceção — não avisa em log e segue."""
    df = pd.DataFrame(columns=['unidades', 'lag_7'])
    with pytest.raises(VazamentoDetectado):
        auditar_whitelist(df, features=['unidades', 'lag_7'])


def test_auditoria_aceita_features_validas():
    df = pd.DataFrame(columns=['lag_1', 'roll_mean_7', 'cal_dow_sin'])
    assert auditar_whitelist(df, features=list(df.columns)) is True


# ===========================================================================
# O que a whitelist NÃO cobre — Capítulo 3.5
# ===========================================================================

def test_lag_mal_construido_passa_pela_whitelist(serie_smooth):
    """
    Documenta a limitação em vez de escondê-la.

    A whitelist valida o NOME da coluna, não o CONTEÚDO. Um lag_1
    construído sem .shift() vaza o futuro inteiro e passa pelo filtro
    com nota máxima. A defesa contra isso é a verificação de conteúdo.
    """
    df = serie_smooth.copy()
    df['lag_1'] = df['unidades']  # ERRO: faltou o .shift(1)

    assert 'lag_1' in montar_feature_cols(df), (
        "a whitelist deveria aprovar — ela só olha o nome"
    )
    assert not verifica_defasagem(df, 'lag_1', 'unidades', lag=1), (
        "a verificação de conteúdo deveria reprovar"
    )


def test_lag_bem_construido_passa_na_verificacao(serie_smooth):
    df = serie_smooth.copy()
    df['lag_1'] = df['unidades'].shift(1)
    assert verifica_defasagem(df, 'lag_1', 'unidades', lag=1)


def test_features_construidas_respeitam_defasagem(serie_smooth):
    """Toda feature lag_N gerada pelo databuilder contém o target defasado."""
    feat = construir_features(serie_smooth)
    for L in (1, 7, 14, 28):
        assert verifica_defasagem(feat, f'lag_{L}', 'unidades', lag=L), (
            f"lag_{L} não corresponde ao target defasado em {L}"
        )


def test_rolling_nao_enxerga_o_presente(serie_smooth):
    """
    A média móvel de hoje não pode incluir o valor de hoje.
    O contrato do Capítulo 3.3: o .shift(1) vem antes da agregação.
    """
    feat = construir_features(serie_smooth)
    esperado = feat['unidades'].shift(1).rolling(7, min_periods=2).mean()
    obtido = feat['roll_mean_7']
    ok = esperado.notna() & obtido.notna()
    assert np.allclose(obtido[ok], esperado[ok]), (
        "roll_mean_7 inclui o valor do próprio período — vazamento"
    )


# ===========================================================================
# Capítulo 3 — causalidade temporal
# ===========================================================================

def test_walk_forward_nunca_treina_no_futuro(serie_smooth):
    folds = list(walk_forward(serie_smooth, n_folds=8, tamanho_fold=14,
                              avisar=False))
    assert len(folds) == 8
    for i, (tr, te) in enumerate(folds):
        assert max(tr) < min(te), f"fold {i}: treino invade o teste"


def test_walk_forward_folds_de_teste_sao_disjuntos(serie_smooth):
    testes = [set(te) for _, te in walk_forward(serie_smooth, n_folds=8,
                                                tamanho_fold=14, avisar=False)]
    for i in range(len(testes)):
        for j in range(i + 1, len(testes)):
            assert not (testes[i] & testes[j]), f"folds {i} e {j} se sobrepõem"


def test_walk_forward_expanding_cresce(serie_smooth):
    tam = [len(tr) for tr, _ in walk_forward(serie_smooth, n_folds=8,
                                             tamanho_fold=14, modo='expanding',
                                             avisar=False)]
    assert tam == sorted(tam) and tam[0] < tam[-1]


def test_walk_forward_sliding_mantem_tamanho(serie_smooth):
    tam = [len(tr) for tr, _ in walk_forward(serie_smooth, n_folds=8,
                                             tamanho_fold=14, modo='sliding',
                                             tamanho_janela=90, avisar=False)]
    assert len(set(tam)) == 1, "janela deslizante mudou de tamanho"


def test_gap_separa_treino_de_teste(serie_smooth):
    gap = 28
    for tr, te in walk_forward(serie_smooth, n_folds=6, tamanho_fold=14,
                               gap=gap, avisar=False):
        assert min(te) - max(tr) > gap, "gap não foi respeitado"


def test_regra_dos_100_emite_alerta(serie_smooth):
    with pytest.warns(UserWarning, match="100"):
        list(walk_forward(serie_smooth, n_folds=3, tamanho_fold=7))


def test_verificar_causalidade_detecta_violacao():
    ruim = [(np.array([0, 1, 2, 5]), np.array([3, 4]))]
    with pytest.raises(AssertionError):
        verificar_causalidade(ruim)


# ===========================================================================
# Métricas — Capítulo 4.3, Caso 5
# ===========================================================================

def test_wmape_formula():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 180.0, 330.0])
    assert np.isclose(wmape(y, p), 60 / 600)


def test_wmape_sobrevive_a_zeros():
    y = np.array([0.0, 0.0, 100.0])
    p = np.array([5.0, 0.0, 90.0])
    assert np.isfinite(wmape(y, p)), "WMAPE explodiu com zeros"


def test_wmape_serie_zerada_retorna_nan():
    assert np.isnan(wmape(np.zeros(5), np.array([1.0, 2, 0, 1, 3])))


def test_wmape_previsao_perfeita_e_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert wmape(y, y) == 0.0


def test_skill_score_sinal():
    assert skill_score_forecast(0.10, 0.20) > 0
    assert skill_score_forecast(0.20, 0.10) < 0
    assert np.isclose(skill_score_forecast(0.10, 0.10), 0.0)


def test_classificar_regime_reconhece_os_quadrantes(dados):
    """Os regimes gerados caem nos quadrantes ADI/CV² esperados."""
    esperado = {'SKU_00': 'Smooth', 'SKU_06': 'Erratic',
                'SKU_11': 'Intermittent', 'SKU_16': 'Lumpy'}
    for sku, cls in esperado.items():
        y = dados[dados.sku == sku]['unidades'].values
        assert classificar_regime(y)['classificacao'] == cls, (
            f"{sku} deveria ser {cls}"
        )


# ===========================================================================
# Comparação estatística — Capítulo 5
# ===========================================================================

def test_bootstrap_detecta_autocorrelacao_e_usa_bloco():
    """Erros autocorrelacionados devem acionar o block bootstrap."""
    rng = np.random.default_rng(0)
    e = np.zeros(300)
    for i in range(1, 300):
        e[i] = 0.85 * e[i - 1] + rng.normal(0, 0.02)
    e += 0.2
    r = bootstrap_ci(e)
    assert r['metodo'] == 'bloco', (
        f"autocorrelação {r['autocorrelacao_lag1']:.2f} não acionou bloco"
    )


def test_bootstrap_usa_classico_quando_independente():
    rng = np.random.default_rng(0)
    assert bootstrap_ci(rng.normal(0.2, 0.03, 300))['metodo'] == 'classico'


def test_bootstrap_ic_contem_a_media():
    rng = np.random.default_rng(0)
    r = bootstrap_ci(rng.normal(0.2, 0.03, 200))
    assert r['li'] < r['media'] < r['ls']


def test_bootstrap_alerta_amostra_pequena():
    with pytest.warns(UserWarning, match="amostra pequena"):
        bootstrap_ci(np.random.default_rng(0).normal(0.2, 0.03, 40))


def test_permutacao_detecta_diferenca_real():
    rng = np.random.default_rng(0)
    b = rng.normal(0.30, 0.02, 150)
    assert teste_permutacao_pareado(b - 0.08, b)['p_valor'] < 0.01


def test_permutacao_nao_inventa_diferenca():
    """Com dois conjuntos equivalentes, p não deve ser pequeno."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.30, 0.02, 150)
    b = rng.normal(0.30, 0.02, 150)
    assert teste_permutacao_pareado(a, b)['p_valor'] > 0.05


def test_bonferroni_ajusta_pelo_numero_de_comparacoes():
    r = comparacao_bonferroni([0.01, 0.04, 0.20], alpha=0.05)
    assert np.isclose(r['p_ajustados'][0], 0.03)
    assert r['significativos'][0]
    assert not r['significativos'][1], "0.04 x 3 = 0.12 não é significativo"
    assert np.isclose(r['alpha_efetivo'], 0.05 / 3)


def test_bonferroni_nunca_passa_de_um():
    r = comparacao_bonferroni([0.5, 0.6], alpha=0.05)
    assert (r['p_ajustados'] <= 1.0).all()


# ===========================================================================
# Gate F3 — Capítulo 6.2
# ===========================================================================

def test_gate_aprova_modelo_consistentemente_melhor():
    rng = np.random.default_rng(1)
    base = rng.normal(0.30, 0.02, 150)
    assert gate_f3(base - rng.normal(0.09, 0.01, 150), base)['aprovado']


def test_gate_reprova_modelo_equivalente():
    rng = np.random.default_rng(1)
    base = rng.normal(0.30, 0.02, 150)
    mod = rng.normal(0.30, 0.02, 150)
    assert not gate_f3(mod, base)['aprovado']


def test_gate_reprova_modelo_pior():
    rng = np.random.default_rng(1)
    base = rng.normal(0.20, 0.02, 150)
    r = gate_f3(base + 0.05, base)
    assert not r['aprovado']
    assert r['criterio_2_skill']['skill'] < 0


def test_gate_reprova_ganho_pequeno_demais():
    """Diferença estatisticamente significativa mas irrelevante na prática."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.30, 0.005, 400)
    r = gate_f3(base - 0.004, base)
    assert r['criterio_1_significancia']['passou']
    assert not r['criterio_2_skill']['passou']
    assert not r['aprovado'], "ganho de ~1% não deveria aprovar"


def test_bonferroni_endurece_o_gate():
    """Mais comparações, mesmo dado: o gate fica mais difícil."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.30, 0.05, 120)
    mod = base - rng.normal(0.03, 0.02, 120)
    p1 = gate_f3(mod, base, n_comparacoes=1)['criterio_1_significancia']['p_ajustado']
    p10 = gate_f3(mod, base, n_comparacoes=10)['criterio_1_significancia']['p_ajustado']
    assert p10 >= p1


# ===========================================================================
# Target encoding — Capítulo 4.3, Caso 6
# ===========================================================================

def test_target_encoding_usa_apenas_o_treino():
    """
    O encoding aplicado ao teste precisa vir das estatísticas do treino.
    Calcular sobre o dataset inteiro é vazamento.
    """
    treino = pd.DataFrame({'cat': ['A'] * 10 + ['B'] * 10,
                           'y': [10.0] * 10 + [20.0] * 10})
    teste = pd.DataFrame({'cat': ['A', 'B'], 'y': [999.0, 999.0]})
    out = target_encoding_seguro(treino, teste, 'cat', 'y', suavizacao=0)
    col = 'cat_cat_mean_smooth'
    assert np.isclose(out[col].iloc[0], 10.0)
    assert np.isclose(out[col].iloc[1], 20.0), (
        "o valor 999 do teste contaminou o encoding"
    )


def test_target_encoding_categoria_nova_cai_na_media_global():
    treino = pd.DataFrame({'cat': ['A'] * 10, 'y': [10.0] * 10})
    teste = pd.DataFrame({'cat': ['Z'], 'y': [0.0]})
    out = target_encoding_seguro(treino, teste, 'cat', 'y')
    assert np.isclose(out['cat_cat_mean_smooth'].iloc[0], 10.0)


# ===========================================================================
# O teste do y embaralhado — Capítulo 1.1, sintoma 5
# ===========================================================================

def test_y_embaralhado_derruba_o_modelo(serie_smooth):
    """
    Com o target permutado aleatoriamente, nenhum modelo honesto pode
    bater o baseline. Se bater, há vazamento generalizado no pipeline.

    Este é o diagnóstico mais poderoso do Capítulo 1, e é automatizável.
    Vale rodar em CI para todo pipeline novo.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    feat = construir_features(serie_smooth)
    cols = montar_feature_cols(feat)
    feat = feat.dropna(subset=cols + ['unidades']).reset_index(drop=True)

    y = rng.permutation(feat['unidades'].values)
    X = feat[cols].values
    corte = int(len(feat) * 0.7)

    modelo = HistGradientBoostingRegressor(max_iter=80, random_state=0)
    modelo.fit(X[:corte], y[:corte])
    pred = np.maximum(modelo.predict(X[corte:]), 0)

    skill = skill_score_forecast(
        wmape(y[corte:], pred),
        wmape(y[corte:], np.full(len(y) - corte, y[:corte].mean())),
    )
    assert skill < 0.05, (
        f"modelo obteve skill {skill:.3f} sobre target ALEATÓRIO — "
        f"isso indica vazamento (Capítulo 1.1, sintoma 5)"
    )


def test_pipeline_completo_nao_vaza(serie_smooth):
    """Teste de integração: o caminho inteiro respeita as regras."""
    feat = construir_features(serie_smooth)
    cols = montar_feature_cols(feat)
    auditar_whitelist(feat, cols)
    feat = feat.dropna(subset=cols + ['unidades']).reset_index(drop=True)
    folds = list(walk_forward(feat, n_folds=6, tamanho_fold=14, gap=28,
                              avisar=False))
    verificar_causalidade(folds)
    assert len(folds) == 6
    assert 'unidades' not in cols
