## Solução da Equipe TANGO

Escolhemos testar diferentes modelos e tratamentos de dados para comparar os desempenhos obtidos. Por isso, nossa solução está dividida em 2 arquivos:

- [Arquivo 1 - XGBoost e ExtraTrees](https://github.com/AbraaoCG/PS-2025.1_tango/blob/main/Solucao_equipe_Tango/draft.ipynb)
- [Arquivo 2 - RandomForest](https://github.com/AbraaoCG/PS-2025.1_tango/blob/main/Solucao_equipe_Tango/Desafio_Analytica_Colab.ipynb)



## Arquivo 1 - XGBoost e ExtraTrees

Tentamos a abordagem de identificar os perfis de jogada "Ofensivo" vs "Defensivo", para estudar a relação do perfil de jogada com o resultado do jogo. Para isso, catalogamos manualmente as posições (colunas Posição 1 e Posição 2) quanto às suas porcentagens de perfil ofensivo. Usamos conhecimento externo ao dataset, com apoio do ChatGPT para chegar a uma porcentagem descritiva de cada posição.

Na remoção de outliers, selecionamos apenas algumas colunas para tratamento, e, para cada uma delas, aplicamos IQR com parâmetros personalizados. Essa abordagem buscou minimizar a eliminação de partidas que, apesar dos outliers de uma coluna, pudessem ser explicadas pelas outras colunas do dataset.

No tratamento de valores nulos, usamos um limiar para que apenas colunas com menos de 50% de valores nulos fossem preenchidas pela mediana. Nosso objetivo foi minimizar o descarte de colunas, ao não desconsiderar colunas com dados nulos, e minimizar o viés de preenchimento, ao não preencher a mediana de dados escassos.

Tentamos criar duas colunas que expressassem a eficiência de ataque de cada time ao relacionar a posição, chutes a gol, escanteios e posse de bola, para ter uma métrica de aproveitamento da posse de bola.

Para selecionar as features usadas nos modelos, exploramos o uso de Recursive Feature Elimination (RFE). Esse método seleciona as features mais importantes, e como resultado disso, descobrimos que as colunas que criamos para mapear padrão ofensivo não foram muito significativas para a predição.

Como modelo escolhemos o ExtraTreesClassifier e o XGBoost.


## Arquivo 2 - RandomForest

Nesse arquivo, exploramos o tratamento de dados com eliminação de colunas com mais de 70% dos dados nulos. Além disso, preenchemos todos os dados nulos restantes com a mediana, para variáveis numéricas, e moda, para variáveis categóricas.

Removemos outliers de todas as colunas usando os mesmos parâmetros de IQR para todas.

Para seleção de features, analisamos similaridade das features com Pearson, ANOVA e visualização de dados, para escolher manualmente as melhores.

O modelo escolhido para esta abordagem foi o RandomForest.


## Comparação dos modelos

Escolhemos os modelos que melhor funcionam para classificação. Para saber qual dos três perfomava melhor comparamos o tempo de execução e as métricas de desempenho com dados balanceados vs desbalanceados.

### Desempenho geral

|Modelo|Tratamento de Nulos|Tratamento de Outliers| Quantidade de features selecionadas|  Tempo de 1 treinamento|
|---|---|---|---|---|
|ExtraTrees|Sem descarte, preenchimento parcial com mediana|IQR em apenas algumas colunas|10| 0.39s |
|XGBoost|Sem descarte, preenchimento parcial com mediana|IQR em apenas algumas colunas|10|0.15s|
|RandomForest|Com descarte, preenchimento com mediana e moda|IQR em todas as colunas|14|1.57s|

### Desempenho com balanceamento das classes

|Modelo| Acurácia Média| Precisão Média | Recall Médio | Média da Aréa ROC |
|---|---|---|---|---|
|ExtraTrees| 69% | 69% | 70% | 77% |
|XGBoost| 69% | 69% | 70% | 77%|
|RandomForest| 69% | 69% | 69% |76% |


### Desempenho sem balanceamento das classes

|Modelo| Acurácia Média| Precisão Média | Recall Médio | Média da Aréa ROC |
|---|---|---|---|---|
|ExtraTrees| 70% | 71% | 77% | 77% |
|XGBoost| 70% | 71% | 76% | 77%|
|RandomForest| 69% | 72% | 73% |76% |