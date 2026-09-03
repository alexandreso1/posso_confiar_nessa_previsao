"""
Validação temporal: walk-forward.

Capítulo 3.2 e 3.4.

A regra que organiza tudo: todo ponto de treino é anterior a todo ponto de
teste, em todos os folds, sem exceção. Não é convenção — é a definição do
que significa validar um modelo que vai prever o futuro.
"""

import warnings

import numpy as np


MINIMO_PONTOS_TESTE = 100


def walk_forward(df, n_folds=10, tamanho_fold=30, modo='expanding',
                 tamanho_janela=None, gap=0, avisar=True):
    """
    Gera pares (indices_treino, indices_teste) para validação temporal.

    Parâmetros
    ----------
    n_folds : número de folds
    tamanho_fold : pontos de teste por fold (o horizonte de previsão)
    modo : 'expanding' (treino cresce) ou 'sliding' (janela fixa)
    tamanho_janela : usado só em modo sliding
    gap : pontos descartados entre treino e teste. Use quando a feature
          mais longa olha para trás — evita que o fim do treino e o começo
          do teste compartilhem informação.
    avisar : emite UserWarning se o total de pontos de teste for menor que
             MINIMO_PONTOS_TESTE (Capítulo 3.2)

    Rende
    -----
    (treino_idx, teste_idx) : arrays de índices posicionais
    """
    n = len(df)
    total_teste = n_folds * tamanho_fold

    if avisar and total_teste < MINIMO_PONTOS_TESTE:
        warnings.warn(
            f"Total de pontos de teste = {total_teste} "
            f"({n_folds} folds x {tamanho_fold}), abaixo do mínimo "
            f"recomendado de {MINIMO_PONTOS_TESTE}. O bootstrap não "
            f"converge bem nessa faixa e o intervalo de confiança será "
            f"largo demais para sustentar decisão. Ver Capítulo 3.2.",
            UserWarning,
            stacklevel=2,
        )

    if modo == 'sliding' and tamanho_janela is None:
        tamanho_janela = max(tamanho_fold * 3, 60)

    inicio_primeiro_teste = n - total_teste
    if inicio_primeiro_teste <= gap:
        raise ValueError(
            f"Série de {n} pontos não comporta {n_folds} folds de "
            f"{tamanho_fold} pontos com gap={gap}. "
            f"Reduza n_folds ou tamanho_fold."
        )

    for k in range(n_folds):
        ini_teste = inicio_primeiro_teste + k * tamanho_fold
        fim_teste = ini_teste + tamanho_fold
        fim_treino = ini_teste - gap

        if modo == 'expanding':
            ini_treino = 0
        else:
            ini_treino = max(0, fim_treino - tamanho_janela)

        treino_idx = np.arange(ini_treino, fim_treino)
        teste_idx = np.arange(ini_teste, fim_teste)

        if len(treino_idx) == 0:
            continue
        yield treino_idx, teste_idx


def verificar_causalidade(folds):
    """
    Confirma que nenhum fold treina no futuro do que testa.

    Deveria ser redundante — walk_forward garante isso por construção.
    Existe porque garantia por construção que ninguém verifica é fé, não
    engenharia.
    """
    for i, (tr, te) in enumerate(folds):
        if len(tr) and len(te) and max(tr) >= min(te):
            raise AssertionError(
                f"Fold {i}: treino vai até o índice {max(tr)} e o teste "
                f"começa em {min(te)} — vazamento temporal."
            )
    return True
