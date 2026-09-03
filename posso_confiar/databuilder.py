"""
Construção de features e defesas contra vazamento.

Capítulo 4.1 e 4.2.

A ideia central: a lista de features é definida por INCLUSÃO (whitelist de
prefixos), nunca por exclusão. Exclusão exige lembrar de tudo que não pode
entrar; inclusão exige declarar o que pode. A primeira falha em silêncio
quando uma coluna nova aparece no dataset; a segunda não.
"""

import warnings

import numpy as np
import pandas as pd


PREFIXOS_PERMITIDOS = [
    'lag_',     # lags simples: lag_1, lag_7, lag_30
    'roll_',    # janelas móveis: roll_mean_7, roll_max_14
    'ewma_',    # médias exponenciais: ewma_alpha02
    'trend_',   # tendência local: trend_slope_30
    'cal_',     # calendário: cal_dow_sin, cal_month_cos
    'cat_',     # target encoding por categoria: cat_sku_mean_smooth
    'ext_',     # exógenas validadas: ext_promo_flag
]


class VazamentoDetectado(Exception):
    """Levantada quando uma coluna proibida chega às features."""


def montar_feature_cols(df, prefixos=None):
    """
    Retorna as colunas do DataFrame elegíveis como feature.

    Apenas colunas cujo nome comece com um dos prefixos permitidos.
    Qualquer coluna original do dataset — target, identificadores, datas —
    fica de fora por construção.
    """
    prefixos = prefixos or PREFIXOS_PERMITIDOS
    return [c for c in df.columns if any(c.startswith(p) for p in prefixos)]


def auditar_whitelist(df, features, prefixos=None):
    """
    Verifica que toda feature declarada respeita a whitelist.

    Falha barulhento: levanta exceção em vez de avisar em log. O custo de
    parar o pipeline é uma manhã de debug; o custo de seguir com vazamento
    é um modelo aprovado por engano.
    """
    prefixos = prefixos or PREFIXOS_PERMITIDOS
    intrusas = [c for c in features
                if not any(c.startswith(p) for p in prefixos)]
    if intrusas:
        raise VazamentoDetectado(
            f"Colunas fora da whitelist chegaram às features: {intrusas}\n"
            f"Prefixos permitidos: {prefixos}"
        )
    return True


def verifica_defasagem(df, col_feature, col_target, lag, tol=1e-9):
    """
    Verifica que a feature realmente contém o target defasado.

    Esta é a verificação de CONTEÚDO, que complementa a whitelist
    (verificação de NOME). Um `lag_1` construído sem .shift() tem nome
    correto e conteúdo vazado — a whitelist aprova, esta função reprova.

    Ver Princípio 3, Capítulo 3.5.
    """
    esperado = df[col_target].shift(lag)
    obtido = df[col_feature]
    comparavel = esperado.notna() & obtido.notna()
    if comparavel.sum() == 0:
        return False
    return bool(np.allclose(obtido[comparavel], esperado[comparavel], atol=tol))


def construir_features(df, col_target='unidades', col_data='data',
                       lags=(1, 7, 14, 28), janelas=(7, 28)):
    """
    Constrói o conjunto de features de uma série única.

    Todas as features derivadas do target passam por .shift(1) antes de
    qualquer agregação — o contrato do Capítulo 3.3. A janela móvel de
    hoje não pode enxergar o valor de hoje.
    """
    out = df.copy().sort_values(col_data).reset_index(drop=True)
    y = out[col_target]

    # o shift(1) vem PRIMEIRO. Tudo abaixo deriva da série já defasada.
    y_passado = y.shift(1)

    for L in lags:
        out[f'lag_{L}'] = y.shift(L)

    for J in janelas:
        out[f'roll_mean_{J}'] = y_passado.rolling(J, min_periods=2).mean()
        out[f'roll_std_{J}'] = y_passado.rolling(J, min_periods=2).std()

    out['ewma_alpha03'] = y_passado.ewm(alpha=0.3, min_periods=2).mean()

    # calendário: derivado da data, nunca do target
    dow = out[col_data].dt.dayofweek
    out['cal_dow_sin'] = np.sin(2 * np.pi * dow / 7)
    out['cal_dow_cos'] = np.cos(2 * np.pi * dow / 7)
    out['cal_dia_mes'] = out[col_data].dt.day

    # exógenas: disponíveis no momento da previsão
    if 'flag_promocao' in out.columns:
        out['ext_promo'] = out['flag_promocao']
    if 'temperatura_media' in out.columns:
        out['ext_temperatura'] = out['temperatura_media']
    if 'preco' in out.columns:
        out['ext_preco'] = out['preco']

    return out


def target_encoding_seguro(df_treino, df_aplicar, col_categoria,
                           col_target, suavizacao=10):
    """
    Target encoding calculado APENAS sobre o treino do fold.

    Calcular a média por categoria sobre o dataset inteiro é o Caso 6 do
    Capítulo 4.3 — vazamento clássico, porque a média usada para prever o
    fold N incorpora informação dos folds N+1 em diante.
    """
    media_global = df_treino[col_target].mean()
    agg = df_treino.groupby(col_categoria)[col_target].agg(['mean', 'count'])

    # suavização bayesiana: categorias raras puxam para a média global
    encoding = (
        (agg['mean'] * agg['count'] + media_global * suavizacao)
        / (agg['count'] + suavizacao)
    )

    col = f'cat_{col_categoria}_mean_smooth'
    saida = df_aplicar.copy()
    saida[col] = saida[col_categoria].map(encoding).fillna(media_global)
    return saida
