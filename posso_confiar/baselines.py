"""
Baselines.

Capítulo 2.3.

Um baseline não é o modelo que você espera perder. É o teto provisório que
o modelo candidato precisa superar para justificar sua existência —
Princípio 2. Rodar baseline primeiro é o que impede seis meses de trabalho
num modelo que nunca bateu a média móvel.
"""

import numpy as np
import pandas as pd


def naive(y_treino, h):
    """Repete o último valor observado."""
    return np.full(h, y_treino[-1], dtype=float)


def naive_sazonal(y_treino, h, periodo=7):
    """Repete o valor do mesmo dia do ciclo anterior."""
    if len(y_treino) < periodo:
        return naive(y_treino, h)
    ultimo_ciclo = y_treino[-periodo:]
    return np.array([ultimo_ciclo[i % periodo] for i in range(h)], dtype=float)


def media_movel(y_treino, h, janela=28):
    """Média dos últimos `janela` pontos."""
    j = min(janela, len(y_treino))
    return np.full(h, float(np.mean(y_treino[-j:])))


def drift(y_treino, h):
    """Extrapola a reta que liga o primeiro ao último ponto."""
    n = len(y_treino)
    if n < 2:
        return naive(y_treino, h)
    inclinacao = (y_treino[-1] - y_treino[0]) / (n - 1)
    return y_treino[-1] + inclinacao * np.arange(1, h + 1)


def croston(y_treino, h, alpha=0.1):
    """
    Método de Croston (1972) para demanda intermitente.

    Estima separadamente o tamanho da demanda e o intervalo entre
    demandas, e prevê a razão entre os dois. Ver Capítulo 7.2.
    """
    y = np.asarray(y_treino, dtype=float)
    nz = np.flatnonzero(y > 0)
    if len(nz) < 2:
        return np.full(h, y.mean() if len(y) else 0.0)

    tamanhos = y[nz]
    intervalos = np.diff(np.concatenate([[-1], nz]))

    z = tamanhos[0]
    x = float(intervalos[0]) if intervalos[0] > 0 else 1.0
    for i in range(1, len(tamanhos)):
        z = alpha * tamanhos[i] + (1 - alpha) * z
        x = alpha * intervalos[i] + (1 - alpha) * x

    return np.full(h, z / max(x, 1e-9))


def sba(y_treino, h, alpha=0.1):
    """
    Syntetos-Boylan Approximation (2001): Croston com correção de viés.

    Croston é enviesado para cima; SBA multiplica por (1 - alpha/2).
    """
    return croston(y_treino, h, alpha) * (1 - alpha / 2)


CATALOGO = {
    'naive': naive,
    'naive_sazonal': naive_sazonal,
    'media_movel': media_movel,
    'drift': drift,
    'croston': croston,
    'sba': sba,
}


def melhor_baseline(y_treino, y_teste, metrica, candidatos=None):
    """
    Roda todos os baselines e devolve o melhor.

    O gate F3 compara o modelo contra o MELHOR baseline, não contra o mais
    conveniente. Escolher um baseline fraco de propósito é a forma mais
    fácil de aprovar um modelo ruim.
    """
    candidatos = candidatos or list(CATALOGO)
    h = len(y_teste)
    resultados = {}
    for nome in candidatos:
        pred = CATALOGO[nome](y_treino, h)
        erro = metrica(y_teste, pred)
        if not np.isnan(erro):
            resultados[nome] = erro
    if not resultados:
        return None, np.nan, {}
    nome_melhor = min(resultados, key=resultados.get)
    return nome_melhor, resultados[nome_melhor], resultados
