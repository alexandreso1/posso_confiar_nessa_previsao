"""
Case completo: do dataset bruto ao veredito do gate F3.

Percorre, em ordem, o método defendido no livro:

    1. Diagnosticar o regime da série            (Capítulo 7.1)
    2. Estabelecer o baseline                    (Capítulo 2)
    3. Construir features com whitelist          (Capítulo 4)
    4. Validar com walk-forward                  (Capítulo 3)
    5. Comparar com rigor estatístico            (Capítulo 5)
    6. Decidir com o gate F3                     (Capítulo 6)
    7. Monitorar estabilidade                    (Capítulo 8)

Rodar:
    python case/case_completo.py

O resultado esperado NÃO é "modelo aprovado em tudo". É que alguns SKUs
aprovem e outros não — e que a razão de cada reprovação seja legível.
Reprovação informativa é o produto do livro.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posso_confiar.baselines import melhor_baseline
from posso_confiar.comparacao import bootstrap_ci, gate_f3
from posso_confiar.dados import gerar_dataset
from posso_confiar.databuilder import (
    auditar_whitelist, construir_features, montar_feature_cols,
)
from posso_confiar.metricas import classificar_regime, skill_score_forecast, wmape
from posso_confiar.validation import verificar_causalidade, walk_forward


N_FOLDS = 12
TAMANHO_FOLD = 14
GAP = 28  # maior lag usado, para não colar treino e teste


def avaliar_sku(df_sku, verbose=False):
    """Roda o método completo para um SKU e devolve o veredito."""
    y_bruto = df_sku['unidades'].values

    # ---- 1. regime -------------------------------------------------------
    regime = classificar_regime(y_bruto)

    # ---- 3. features (whitelist por inclusão) ---------------------------
    feat = construir_features(df_sku)
    cols = montar_feature_cols(feat)
    auditar_whitelist(feat, cols)

    feat = feat.dropna(subset=cols + ['unidades']).reset_index(drop=True)
    if len(feat) < N_FOLDS * TAMANHO_FOLD + GAP + 60:
        return dict(sku=df_sku['sku'].iloc[0], regime=regime,
                    status='SERIE_CURTA', n=len(feat))

    X = feat[cols].values
    y = feat['unidades'].values

    # ---- 4. walk-forward -------------------------------------------------
    folds = list(walk_forward(feat, n_folds=N_FOLDS, tamanho_fold=TAMANHO_FOLD,
                              gap=GAP, avisar=False))
    verificar_causalidade(folds)

    erros_modelo, erros_baseline, nomes_baseline = [], [], []

    for treino_idx, teste_idx in folds:
        Xtr, ytr = X[treino_idx], y[treino_idx]
        Xte, yte = X[teste_idx], y[teste_idx]

        # ---- 2. baseline (o melhor, não o conveniente) ------------------
        nome_b, erro_b, _ = melhor_baseline(ytr, yte, wmape)
        if nome_b is None:
            continue

        # ---- modelo candidato -------------------------------------------
        modelo = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.08, max_depth=4, random_state=0,
        )
        modelo.fit(Xtr, ytr)
        pred = np.maximum(modelo.predict(Xte), 0)
        erro_m = wmape(yte, pred)

        if np.isnan(erro_m) or np.isnan(erro_b):
            continue
        erros_modelo.append(erro_m)
        erros_baseline.append(erro_b)
        nomes_baseline.append(nome_b)

    if len(erros_modelo) < 3:
        return dict(sku=df_sku['sku'].iloc[0], regime=regime,
                    status='DADOS_INSUFICIENTES', n_folds=len(erros_modelo))

    # ---- 5 e 6. comparação e gate ---------------------------------------
    resultado = gate_f3(erros_modelo, erros_baseline, n_comparacoes=1)

    # ---- 7. estabilidade entre folds ------------------------------------
    cv_erro = float(np.std(erros_modelo) / np.mean(erros_modelo))

    return dict(
        sku=df_sku['sku'].iloc[0],
        regime=regime,
        status='AVALIADO',
        wmape_modelo=float(np.mean(erros_modelo)),
        wmape_baseline=float(np.mean(erros_baseline)),
        baseline_vencedor=pd.Series(nomes_baseline).mode()[0],
        skill=resultado['criterio_2_skill']['skill'],
        p_valor=resultado['criterio_1_significancia']['p_ajustado'],
        metodo_bootstrap=resultado['criterio_3_intervalos']['metodo_bootstrap'],
        autocorrelacao=resultado['criterio_3_intervalos']['autocorrelacao'],
        cv_entre_folds=cv_erro,
        aprovado=resultado['aprovado'],
        detalhe=resultado,
    )


def main():
    print("=" * 78)
    print("  CASE COMPLETO — Posso confiar nessa previsão?")
    print("=" * 78)

    caminho = Path(__file__).resolve().parent.parent / 'dados' / 'demanda_sintetica.csv'
    if caminho.exists():
        df = pd.read_csv(caminho, parse_dates=['data'])
        print(f"\nDataset carregado de {caminho.name}")
    else:
        df = gerar_dataset()
        print("\nDataset gerado em memória")

    print(f"  {len(df):,} linhas | {df['sku'].nunique()} SKUs | "
          f"{df['data'].min().date()} a {df['data'].max().date()}")

    print(f"\nConfiguração da validação:")
    print(f"  {N_FOLDS} folds x {TAMANHO_FOLD} pontos = "
          f"{N_FOLDS * TAMANHO_FOLD} pontos de teste por SKU")
    print(f"  gap de {GAP} pontos entre treino e teste (maior lag usado)")

    resultados = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sku, g in df.groupby('sku'):
            resultados.append(avaliar_sku(g.reset_index(drop=True)))

    avaliados = [r for r in resultados if r['status'] == 'AVALIADO']
    pulados = [r for r in resultados if r['status'] != 'AVALIADO']

    print("\n" + "=" * 78)
    print("  RESULTADO POR SKU")
    print("=" * 78)
    print(f"\n{'SKU':8} {'regime':14} {'WMAPE mod':>10} {'WMAPE base':>11} "
          f"{'skill':>7} {'p':>7} {'gate':>10}")
    print("-" * 78)
    for r in sorted(avaliados, key=lambda x: -x['skill']):
        print(f"{r['sku']:8} {r['regime']['classificacao']:14} "
              f"{r['wmape_modelo']:>10.3f} {r['wmape_baseline']:>11.3f} "
              f"{r['skill']:>7.3f} {r['p_valor']:>7.4f} "
              f"{'APROVADO' if r['aprovado'] else 'reprovado':>10}")

    for r in pulados:
        print(f"{r['sku']:8} {r['regime']['classificacao']:14} "
              f"{'—':>10} {'—':>11} {'—':>7} {'—':>7} {r['status']:>10}")

    aprovados = [r for r in avaliados if r['aprovado']]
    print("\n" + "=" * 78)
    print("  LEITURA DOS RESULTADOS")
    print("=" * 78)
    print(f"\n  Avaliados: {len(avaliados)} | aprovados no gate F3: "
          f"{len(aprovados)} | reprovados: {len(avaliados) - len(aprovados)}")
    if pulados:
        print(f"  Não avaliados: {len(pulados)} "
              f"({', '.join(r['sku'] for r in pulados)})")

    print("\n  Por regime:")
    por_regime = {}
    for r in avaliados:
        c = r['regime']['classificacao']
        por_regime.setdefault(c, []).append(r['aprovado'])
    for c, lista in sorted(por_regime.items()):
        print(f"    {c:14} {sum(lista)}/{len(lista)} aprovados")

    print("\n  Motivo das reprovações:")
    for r in avaliados:
        if r['aprovado']:
            continue
        d = r['detalhe']
        falhas = []
        if not d['criterio_1_significancia']['passou']:
            falhas.append(f"p={d['criterio_1_significancia']['p_ajustado']:.3f}")
        if not d['criterio_2_skill']['passou']:
            falhas.append(f"skill={d['criterio_2_skill']['skill']:.3f}")
        if not d['criterio_3_intervalos']['passou']:
            falhas.append("IC sobrepostos")
        print(f"    {r['sku']:8} {', '.join(falhas)}")

    n_bloco = sum(1 for r in avaliados if r['metodo_bootstrap'] == 'bloco')
    print(f"\n  Bootstrap em bloco acionado em {n_bloco} de {len(avaliados)} SKUs")
    print("    (autocorrelação dos erros acima do limiar — Capítulo 5.2)")

    print("\n" + "=" * 78)
    print("""
  O QUE ESTE RESULTADO SIGNIFICA

  Se todos os SKUs tivessem sido aprovados, o gate não estaria fazendo
  trabalho nenhum. O valor está na separação: alguns modelos superam o
  baseline de forma sustentada, outros não — e o motivo de cada reprovação
  é explícito.

  Um modelo reprovado por skill baixo é diferente de um reprovado por
  p-valor: o primeiro não é melhor, o segundo pode ser melhor mas não há
  evidência suficiente para afirmar. As duas conversas com o stakeholder
  são diferentes, e o gate permite ter a conversa certa.

  Ver Capítulo 6.3 — quando NÃO passar é o sinal certo.
""")


if __name__ == '__main__':
    main()
