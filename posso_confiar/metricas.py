"""
Métricas de erro e de regime.

Capítulos 2.1 e 7.1; Apêndice A.

Nota sobre autoria das fórmulas: WMAPE, MASE, skill score, ADI e CV² são
definições publicadas na literatura e creditadas no Apêndice E do livro.
As implementações aqui são traduções diretas dessas fórmulas; o que é
escolha deste livro são as decisões operacionais — tratamento de zeros,
valores default, o que fazer quando a série é degenerada.
"""

import numpy as np


def wmape(y_true, y_pred):
    """
    Weighted Mean Absolute Percentage Error.

    Soma dos erros absolutos sobre soma dos valores reais. É robusta a
    zeros pontuais, ao contrário do MAPE — ver Capítulo 4.3, Caso 5.

    Retorna np.nan se a série for inteiramente zerada, porque nesse caso
    não existe erro percentual definido.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    soma_erros = np.sum(np.abs(y_pred - y_true))
    soma_reais = np.sum(np.abs(y_true))

    if soma_reais == 0:
        return np.nan
    return soma_erros / soma_reais


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred, float) - np.asarray(y_true, float)) ** 2)))


def skill_score_forecast(erro_modelo, erro_baseline):
    """
    Skill score: ganho relativo do modelo sobre o baseline.

    Positivo = modelo melhor. Zero = empate. Negativo = baseline melhor.
    O gate F3 (Capítulo 6.2) exige skill > 0.05.
    """
    if erro_baseline == 0:
        return np.nan
    return 1.0 - (erro_modelo / erro_baseline)


def classificar_regime(serie):
    """
    Classifica a série nos quadrantes ADI/CV² de Syntetos & Boylan (2005).

    ADI  = intervalo médio entre demandas não-nulas
    CV²  = coeficiente de variação ao quadrado dos tamanhos não-nulos

    Cortes canônicos: ADI = 1.32, CV² = 0.49.
    """
    y = np.asarray(serie, dtype=float)
    nao_nulos = y[y > 0]

    if len(nao_nulos) == 0:
        return dict(adi=np.inf, cv2=np.nan, classificacao='Série zerada')
    if len(nao_nulos) == 1:
        return dict(adi=float(len(y)), cv2=0.0, classificacao='Dados insuficientes')

    adi = len(y) / len(nao_nulos)
    cv2 = float((nao_nulos.std() / nao_nulos.mean()) ** 2)

    if adi >= 1.32 and cv2 >= 0.49:
        cls = 'Lumpy'
    elif adi >= 1.32:
        cls = 'Intermittent'
    elif cv2 >= 0.49:
        cls = 'Erratic'
    else:
        cls = 'Smooth'

    return dict(adi=float(adi), cv2=cv2, classificacao=cls)
