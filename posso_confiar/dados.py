"""
Gerador do dataset sintético que acompanha o livro.

O dataset tem 20 SKUs distribuídos pelos quatro quadrantes da taxonomia
ADI/CV² de Syntetos & Boylan (Capítulo 7.1), porque um conjunto de dados
que só tem séries suaves não exercita nada do que o livro ensina.

Há também dois SKUs com armadilhas deliberadas:
    - SKU_18: quebra de regime na metade da série (drift de conceito)
    - SKU_19: série curta demais para o gate F3 (Capítulo 3.2)

O gerador é determinístico: mesma seed, mesmo dataset.
"""

import numpy as np
import pandas as pd


# Configuração dos regimes. ADI = intervalo médio entre demandas,
# CV2 = coeficiente de variação ao quadrado do tamanho da demanda.
# Os cortes de Syntetos-Boylan são ADI = 1.32 e CV2 = 0.49.
REGIMES = {
    'smooth':       dict(p_venda=0.98, cv=0.25, nivel=120),
    'erratic':      dict(p_venda=0.95, cv=0.95, nivel=80),
    'intermittent': dict(p_venda=0.45, cv=0.30, nivel=25),
    'lumpy':        dict(p_venda=0.35, cv=1.10, nivel=40),
}


def _serie_regime(n, regime, rng, com_sazonalidade=True, com_tendencia=True):
    """Gera uma série de demanda diária para um regime dado."""
    cfg = REGIMES[regime]
    t = np.arange(n)

    nivel = np.full(n, float(cfg['nivel']))
    if com_tendencia:
        nivel += cfg['nivel'] * 0.0004 * t

    if com_sazonalidade:
        # sazonalidade semanal: fim de semana mais fraco
        dow = t % 7
        fator_dow = np.where(dow >= 5, 0.65, 1.10)
        # sazonalidade anual suave
        fator_ano = 1 + 0.15 * np.sin(2 * np.pi * t / 365)
        nivel = nivel * fator_dow * fator_ano

    # tamanho da demanda quando há venda: lognormal com CV alvo
    sigma = np.sqrt(np.log(1 + cfg['cv'] ** 2))
    mu = np.log(np.maximum(nivel, 1e-6)) - sigma ** 2 / 2
    tamanho = rng.lognormal(mu, sigma)

    # ocorrência de venda
    ocorre = rng.random(n) < cfg['p_venda']

    return np.where(ocorre, tamanho, 0.0).round(0)


def gerar_dataset(n_dias=730, seed=42):
    """
    Gera o dataset completo.

    Retorna DataFrame com colunas:
        data, sku, categoria, regime_verdadeiro, unidades,
        preco, flag_promocao, temperatura_media
    """
    rng = np.random.default_rng(seed)
    datas = pd.date_range('2023-01-01', periods=n_dias, freq='D')

    # variáveis exógenas compartilhadas
    t = np.arange(n_dias)
    temperatura = 24 + 6 * np.sin(2 * np.pi * (t - 20) / 365) + rng.normal(0, 2, n_dias)
    promocao_geral = (rng.random(n_dias) < 0.08).astype(int)

    plano = (
        [('smooth', 'bebidas')] * 6 +
        [('erratic', 'higiene')] * 5 +
        [('intermittent', 'pecas')] * 5 +
        [('lumpy', 'sazonais')] * 4
    )

    linhas = []
    for i, (regime, categoria) in enumerate(plano):
        sku = f"SKU_{i:02d}"
        y = _serie_regime(n_dias, regime, rng)

        # SKU_18: quebra de regime na metade (drift de conceito)
        if sku == 'SKU_18':
            corte = n_dias // 2
            y[corte:] = _serie_regime(n_dias - corte, 'smooth', rng)[:]
            regime = 'lumpy->smooth'

        # efeito de promoção: eleva a demanda
        y = y * (1 + 0.45 * promocao_geral)

        # efeito de temperatura, só em bebidas
        if categoria == 'bebidas':
            y = y * (1 + 0.02 * (temperatura - temperatura.mean()))

        preco = np.round(
            (8 + 2 * (i % 5)) * (1 - 0.15 * promocao_geral) + rng.normal(0, 0.2, n_dias),
            2,
        )

        n_sku = n_dias
        # SKU_19: série curta deliberadamente (só os últimos 120 dias)
        inicio = 0
        if sku == 'SKU_19':
            inicio = n_dias - 120

        linhas.append(pd.DataFrame({
            'data': datas[inicio:],
            'sku': sku,
            'categoria': categoria,
            'regime_verdadeiro': regime,
            'unidades': np.maximum(y[inicio:], 0).round(0),
            'preco': preco[inicio:],
            'flag_promocao': promocao_geral[inicio:],
            'temperatura_media': temperatura[inicio:].round(1),
        }))

    df = pd.concat(linhas, ignore_index=True)
    return df.sort_values(['sku', 'data']).reset_index(drop=True)


if __name__ == '__main__':
    df = gerar_dataset()
    df.to_csv('dados/demanda_sintetica.csv', index=False)
    print(f"{len(df):,} linhas | {df['sku'].nunique()} SKUs | "
          f"{df['data'].min().date()} a {df['data'].max().date()}")
