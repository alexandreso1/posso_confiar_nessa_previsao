"""
Comparação estatística e o gate F3.

Capítulos 5.2 a 5.4 e 6.2.

Nota sobre o block bootstrap: o bootstrap clássico reamostra pontos
individuais e pressupõe independência. Erros de forecast em folds
consecutivos são autocorrelacionados por construção — folds vizinhos
compartilham quase toda a janela de treino. Reamostrar ponto a ponto
destrói essa estrutura e produz intervalo estreito demais.

A função `bootstrap_ci` abaixo detecta autocorrelação e escolhe o método
apropriado, em vez de deixar a decisão para o leitor lembrar.
"""

import warnings

import numpy as np
from scipy import stats


MINIMO_AMOSTRA = 100
LIMIAR_AUTOCORRELACAO = 0.2


def autocorrelacao_lag1(x):
    """Autocorrelação de lag 1 da amostra."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0.0
    a, b = x[:-1], x[1:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_ci(erros, n_resamples=5000, ci=0.95, seed=42,
                 metodo='auto', tamanho_bloco=None):
    """
    Intervalo de confiança por bootstrap percentil.

    Parâmetros
    ----------
    metodo : 'auto', 'classico' ou 'bloco'
        'auto' mede a autocorrelação de lag 1 e usa block bootstrap se ela
        exceder LIMIAR_AUTOCORRELACAO. É o default porque a decisão certa
        depende dos dados, não da memória de quem chamou a função.
    tamanho_bloco : comprimento do bloco. Se None, usa n^(1/3), a regra
        prática usual para bootstrap em bloco.

    Retorna dict com limites, média, método usado e a autocorrelação medida.
    """
    erros = np.asarray(erros, dtype=float)
    n = len(erros)

    if n < MINIMO_AMOSTRA:
        warnings.warn(
            f"Bootstrap sobre amostra pequena (n={n}, recomendado "
            f">= {MINIMO_AMOSTRA}). O intervalo será reportado, mas tem "
            f"incerteza alta. Ver Capítulo 3.2.",
            UserWarning,
            stacklevel=2,
        )

    rho = autocorrelacao_lag1(erros)

    if metodo == 'auto':
        metodo_usado = 'bloco' if abs(rho) > LIMIAR_AUTOCORRELACAO else 'classico'
    else:
        metodo_usado = metodo

    rng = np.random.default_rng(seed)

    if metodo_usado == 'bloco':
        L = tamanho_bloco or max(2, int(round(n ** (1 / 3))))
        n_blocos = int(np.ceil(n / L))
        inicios_possiveis = max(1, n - L + 1)
        medias = np.empty(n_resamples)
        for i in range(n_resamples):
            inicios = rng.integers(0, inicios_possiveis, size=n_blocos)
            amostra = np.concatenate([erros[s:s + L] for s in inicios])[:n]
            medias[i] = amostra.mean()
    else:
        L = None
        medias = np.array([
            rng.choice(erros, size=n, replace=True).mean()
            for _ in range(n_resamples)
        ])

    alpha = (1 - ci) / 2
    return dict(
        media=float(erros.mean()),
        li=float(np.percentile(medias, 100 * alpha)),
        ls=float(np.percentile(medias, 100 * (1 - alpha))),
        metodo=metodo_usado,
        autocorrelacao_lag1=rho,
        tamanho_bloco=L,
        n=n,
    )


def teste_permutacao_pareado(erros_a, erros_b, n_permutacoes=10000, seed=42):
    """
    Teste de permutação pareado entre dois conjuntos de erros.

    Hipótese nula: os rótulos A e B são intercambiáveis, isto é, não há
    diferença sistemática entre os modelos. A cada permutação, o sinal da
    diferença de cada par é sorteado.

    Pareado porque os dois modelos foram avaliados nos MESMOS folds — o
    que remove a variabilidade entre folds da comparação.
    """
    a = np.asarray(erros_a, dtype=float)
    b = np.asarray(erros_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("erros_a e erros_b precisam ter o mesmo comprimento")

    d = a - b
    observado = d.mean()

    rng = np.random.default_rng(seed)
    sinais = rng.choice([-1.0, 1.0], size=(n_permutacoes, len(d)))
    distribuicao = (sinais * d).mean(axis=1)

    p = float((np.abs(distribuicao) >= abs(observado)).mean())
    return dict(
        diferenca_media=float(observado),
        p_valor=p,
        n_pares=len(d),
    )


def comparacao_bonferroni(p_valores, alpha=0.05):
    """
    Correção de Bonferroni para múltiplas comparações.

    Multiplica cada p-valor pelo número de comparações. Conservador por
    escolha: controla a probabilidade de QUALQUER falso positivo entre
    todas as comparações, não a proporção deles.

    Alternativa menos conservadora, para quem entende a diferença entre
    FWER e FDR: Benjamini-Hochberg (1995), no Apêndice E.
    """
    p = np.asarray(p_valores, dtype=float)
    m = len(p)
    ajustados = np.minimum(p * m, 1.0)
    return dict(
        p_ajustados=ajustados,
        significativos=ajustados < alpha,
        alpha_efetivo=alpha / m,
        n_comparacoes=m,
    )


def gate_f3(erros_modelo, erros_baseline, alpha=0.05, skill_minimo=0.05,
            n_comparacoes=1, seed=42):
    """
    O gate F3 (Capítulo 6.2).

    Três critérios, todos obrigatórios:
      1. p-valor do teste pareado < alpha, corrigido por Bonferroni
      2. skill score > skill_minimo
      3. IC95 dos dois modelos sem sobreposição

    Retorna dict com o veredito e o detalhe de cada critério, para que a
    reprovação seja tão informativa quanto a aprovação.
    """
    em = np.asarray(erros_modelo, dtype=float)
    eb = np.asarray(erros_baseline, dtype=float)

    teste = teste_permutacao_pareado(em, eb, seed=seed)
    p_ajustado = min(teste['p_valor'] * n_comparacoes, 1.0)

    skill = 1.0 - (em.mean() / eb.mean()) if eb.mean() != 0 else np.nan

    ci_m = bootstrap_ci(em, seed=seed)
    ci_b = bootstrap_ci(eb, seed=seed)
    sem_sobreposicao = ci_m['ls'] < ci_b['li']

    c1 = bool(p_ajustado < alpha)
    c2 = bool(skill > skill_minimo)
    c3 = bool(sem_sobreposicao)

    return dict(
        aprovado=c1 and c2 and c3,
        criterio_1_significancia=dict(
            passou=c1, p_bruto=teste['p_valor'], p_ajustado=p_ajustado,
            alpha=alpha, n_comparacoes=n_comparacoes,
        ),
        criterio_2_skill=dict(
            passou=c2, skill=float(skill), minimo=skill_minimo,
        ),
        criterio_3_intervalos=dict(
            passou=c3,
            ic_modelo=(ci_m['li'], ci_m['ls']),
            ic_baseline=(ci_b['li'], ci_b['ls']),
            metodo_bootstrap=ci_m['metodo'],
            autocorrelacao=ci_m['autocorrelacao_lag1'],
        ),
    )
