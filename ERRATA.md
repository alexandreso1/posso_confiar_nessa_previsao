# Errata

Correções e esclarecimentos ao livro identificados após a publicação.

Encontrou algo? Abra uma issue.

---

## 1ª edição (2026)

### Capítulo 5.2 — bootstrap e autocorrelação

**Esclarecimento.** O livro apresenta `bootstrap_ci` com reamostragem clássica e alerta, em ressalva, que ela pressupõe independência entre as observações — premissa que erros de folds consecutivos violam, já que folds vizinhos compartilham quase toda a janela de treino.

A ressalva está correta, mas o livro afirma que "para a maior parte dos projetos de forecast com folds bem-espaçados, bootstrap clássico funciona bem" sem oferecer critério para decidir quando isso vale. Num livro que defende critérios objetivos, a afirmação deveria ser verificável.

**Neste repositório**, `bootstrap_ci` mede a autocorrelação de lag 1 da série de erros e usa block bootstrap quando ela excede 0,2. O método escolhido e a autocorrelação medida vêm no retorno da função.

Rodando o case completo, o block bootstrap é acionado em 8 dos 19 SKUs — ou seja, a situação não é rara.

### Capítulo 6.2 — o critério de intervalos domina o gate

**Observação empírica.** Rodando o gate F3 sobre os 20 SKUs do dataset, o critério 3 (IC95 sem sobreposição) aparece em **todas** as 17 reprovações, incluindo quatro SKUs que passaram nos critérios 1 e 2.

Isso confirma uma tensão que o próprio livro registra: o Bug 9 do Apêndice B observa que comparar por sobreposição de intervalos é mais rigoroso que o teste pareado. Somar os dois critérios torna o gate mais conservador do que o texto do Capítulo 6.2 sugere.

O gate continua defensável — o custo de aprovar um modelo ruim é alto —, mas quem o adotar deve saber que o critério 3 é o que mais reprova na prática, e considerar afrouxá-lo em contextos onde o custo do erro tipo II seja relevante.

### Capítulo 4.2 — o alcance da whitelist

**Esclarecimento.** A whitelist por prefixo valida o **nome** da coluna, não o **conteúdo**. Um `lag_7` construído sem `.shift()` passa pelo filtro e vaza o futuro.

O livro trata da construção correta no Capítulo 3.3 e da whitelist no 4.2, mas não articula explicitamente que são duas camadas distintas de defesa. Quem ler apenas o Capítulo 4 pode concluir que o prefixo basta.

Neste repositório, `verifica_defasagem()` faz a verificação de conteúdo, e o teste `test_lag_mal_construido_passa_pela_whitelist` documenta o caso.
