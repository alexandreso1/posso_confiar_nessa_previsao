# Posso confiar nessa previsão?

Repositório complementar do livro **_Posso confiar nessa previsão? Validação, estabilidade e decisão em forecast de demanda_**, de Alexandre S. Oliveira.

Contém as funções apresentadas no livro em formato executável, um dataset sintético para reproduzir os exemplos, um case completo de ponta a ponta, e os testes automatizados que verificam as propriedades que o livro defende.

Tudo roda com quatro dependências (`pandas`, `numpy`, `scipy`, `scikit-learn`) e sem configuração.

---

## Por que este repositório existe

O livro defende que disciplina precisa estar embutida na arquitetura, não confiada à atenção do praticante. Distribuir código dentro de um ebook — de onde o leitor copia à mão, torcendo para não errar a indentação — seria incoerente com essa tese.

Há uma segunda razão: código em repositório é corrigível. Erro em livro impresso espera a próxima edição.

---

## Começando

```bash
git clone https://github.com/SEU_USUARIO/posso-confiar-nessa-previsao.git
cd posso-confiar-nessa-previsao
pip install -e ".[dev]"

python case/case_completo.py     # o case de ponta a ponta
pytest tests/ -v                 # os 38 testes
```

Python 3.10 ou superior. O case leva cerca de um minuto.

---

## O case completo

`case/case_completo.py` percorre o método do livro sobre 20 SKUs:

1. Diagnosticar o regime da série — Capítulo 7.1
2. Estabelecer o baseline — Capítulo 2
3. Construir features com whitelist — Capítulo 4
4. Validar com walk-forward — Capítulo 3
5. Comparar com rigor estatístico — Capítulo 5
6. Decidir com o gate F3 — Capítulo 6
7. Verificar estabilidade entre folds — Capítulo 8

### O resultado, e por que ele importa

```
SKU      regime          WMAPE mod  WMAPE base   skill       p       gate
------------------------------------------------------------------------
SKU_03   Smooth              0.244       0.308   0.207  0.0124   APROVADO
SKU_02   Smooth              0.240       0.294   0.186  0.0021   APROVADO
SKU_00   Smooth              0.275       0.324   0.152  0.0063  reprovado
...
SKU_11   Intermittent        1.569       1.106  -0.419  0.0100  reprovado
SKU_17   Lumpy               1.978       1.249  -0.584  0.0017  reprovado
SKU_19   Lumpy                   —           —       —       — SERIE_CURTA

Avaliados: 19 | aprovados no gate F3: 2 | reprovados: 17

Por regime:
  Smooth         2/6 aprovados
  Erratic        0/5 aprovados
  Intermittent   0/6 aprovados
  Lumpy          0/2 aprovados
```

**Dois de dezenove.** Se todos passassem, o gate não estaria fazendo trabalho nenhum.

Três leituras que valem a pena:

**O ML perde feio em regime difícil.** Nenhum SKU Erratic, Intermittent ou Lumpy foi aprovado, e o skill é negativo em todos eles — o gradient boosting fica pior que Croston, SBA ou a média móvel. É o Princípio 2 e o Capítulo 7 acontecendo com dados que você pode rodar: **diagnosticar o regime antes de escolher o modelo não é conselho, é economia de meses**.

**O critério de intervalos domina o gate.** "IC sobrepostos" aparece em todas as 17 reprovações, incluindo quatro SKUs Smooth que passaram nos outros dois critérios. Na prática, o terceiro critério do gate F3 é o mais restritivo — o que confirma uma crítica legítima ao desenho do gate e está registrado na [ERRATA](ERRATA.md).

**O block bootstrap é acionado em 8 dos 19 SKUs.** A função `bootstrap_ci` mede a autocorrelação dos erros e troca de método sozinha quando ela passa do limiar, em vez de deixar a decisão para o leitor lembrar.

O SKU_19 é curto de propósito: existe para exercitar o caminho da série que não comporta o gate (Capítulo 3.2).

---

## O dataset

`dados/demanda_sintetica.csv` — 13.990 linhas, 20 SKUs, 730 dias, determinístico.

Os SKUs cobrem os quatro quadrantes da taxonomia ADI/CV² de Syntetos & Boylan, porque um conjunto de dados que só tem séries suaves não exercita nada do que o livro ensina:

| Regime | SKUs | ADI | CV² |
|---|---|---|---|
| Smooth | 00–05 | ~1,0 | ~0,16 |
| Erratic | 06–10 | ~1,1 | ~0,95 |
| Intermittent | 11–15 | ~2,3 | ~0,17 |
| Lumpy | 16, 17, 19 | ~3,0 | ~1,4 |

Mais duas armadilhas deliberadas: **SKU_18** muda de regime na metade da série (drift de conceito) e **SKU_19** tem apenas 120 dias.

Inclui exógenas — preço, flag de promoção e temperatura — para exercitar o Capítulo 7.4. Para regerar com outra seed: `python -m posso_confiar.dados`.

---

## Os testes

Esta é a parte que responde a uma lacuna reconhecida do livro. O Apêndice C oferece um checklist de 24 itens que um humano percorre e assina — e revisão manual falha por fadiga, que é o modo de falha descrito no Capítulo 1. Os itens automatizáveis viram asserções aqui.

```
$ pytest tests/ -q
38 passed in 2.28s
```

Alguns que valem ser lidos:

| Teste | O que verifica |
|---|---|
| `test_whitelist_rejeita_target` | o target não vaza para as features |
| `test_lag_mal_construido_passa_pela_whitelist` | **documenta o que a whitelist não cobre** |
| `test_rolling_nao_enxerga_o_presente` | o `.shift(1)` vem antes da agregação |
| `test_walk_forward_nunca_treina_no_futuro` | causalidade temporal em todos os folds |
| `test_wmape_sobrevive_a_zeros` | a armadilha do Capítulo 4.3, Caso 5 |
| `test_target_encoding_usa_apenas_o_treino` | o encoding não vê o futuro |
| `test_gate_reprova_ganho_pequeno_demais` | significância sem relevância não aprova |
| `test_y_embaralhado_derruba_o_modelo` | o diagnóstico do Capítulo 1.1, sintoma 5 |

Dois merecem destaque.

**`test_lag_mal_construido_passa_pela_whitelist`** constrói um `lag_1` sem `.shift()`, mostra que a whitelist aprova, e mostra que a verificação de conteúdo reprova. A whitelist valida o **nome** da coluna, não o **conteúdo** — e esconder essa limitação seria pior que documentá-la.

**`test_y_embaralhado_derruba_o_modelo`** automatiza o diagnóstico mais poderoso do livro: com o target permutado aleatoriamente, nenhum modelo honesto pode bater o baseline. Se bater, há vazamento. Vale rodar em CI para todo pipeline novo.

---

## Estrutura

```
posso_confiar/
    dados.py          gerador do dataset sintético
    databuilder.py    features, whitelist, auditoria, target encoding
    validation.py     walk-forward, verificação de causalidade
    baselines.py      naive, sazonal, média móvel, drift, Croston, SBA
    metricas.py       WMAPE, skill score, ADI/CV²
    comparacao.py     bootstrap (clássico e em bloco), permutação,
                      Bonferroni, gate F3
tests/
    test_anti_leakage.py
case/
    case_completo.py
dados/
    demanda_sintetica.csv
```

| Módulo | Capítulo |
|---|---|
| `databuilder.py` | 4.1, 4.2, 4.3 |
| `validation.py` | 3.2, 3.3, 3.4 |
| `baselines.py` | 2.3, 7.2 |
| `metricas.py` | 2.1, 7.1, Apêndice A |
| `comparacao.py` | 5.2, 5.3, 5.4, 6.2 |

---

## Uma diferença em relação ao livro

`bootstrap_ci` aqui é mais completa que a versão impressa. O livro apresenta o bootstrap clássico e alerta, em ressalva, que ele pressupõe independência — premissa que erros de folds consecutivos violam.

A versão deste repositório **mede a autocorrelação de lag 1 e troca para block bootstrap** quando ela passa de 0,2. A decisão certa depende dos dados, não da memória de quem chama a função.

```python
r = bootstrap_ci(erros)
r['metodo']               # 'classico' ou 'bloco'
r['autocorrelacao_lag1']  # o que motivou a escolha
```

Ver [ERRATA.md](ERRATA.md).

---

## Errata e contribuições

Correções ao livro identificadas após a publicação ficam em [ERRATA.md](ERRATA.md).

Encontrou um erro? Abra uma issue. Erro reportado é contribuição, não incômodo — é literalmente a tese do livro aplicada a ele mesmo.

---

## Licença

Código sob licença MIT. O texto do livro é protegido por direitos autorais e não está incluído aqui.
