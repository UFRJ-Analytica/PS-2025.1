UFRJ ANALYTICA \- BRAINSTORM  
[Repositório da equipe](http://github.com/Ang3k/Analytica-Desafio)

1 \- Qual o time mais agressivo do campeonato? (Maior número somado de cartões \+ faltas)

2 \- Assertividade do chutes, comparar “Chutes a gol”, “Chutes fora”, “Chutes Bloqueados”, “Defesas difíceis” e “Tiros-livres” comparativamente com os gols marcados.  
(se estivermos comparando a assertividade de chutes do time 1, os dados de “defesas difíceis” e “Chutes bloqueados” devem ser do time 2\)

3 \- Analisar como as estratégias de equipe alteram a taxa de vitória \+ outros.

4 \- Gravidade das faltas, comparar “Faltas”, “Cartões Vermelhos”, “Cartões Amarelos”, “Tratamentos” (se estivermos analisando a gravidade das faltas do time 1, os dados de “Tratamentos” deve ser do time 2\)

5 \- Contra ataques válidos, comparar “impedimentos”, “contra ataques” (?)

6 \- Relação substituições com posse de bola e gols, comparar “Substituições”, “Posse”, Gols”

UFRJ ANALYTICA \- Tratamento de Dados 

1 \- Eliminação / Agregação de Colunas

* Eliminar colunas inúteis pro modelo  
* Misturar colunas que possam ser misturadas  
* Bem Subjetivo

2 \- Remoção de Outliers

* Identificar valores “fora da curva” nas colunas  
* Várias técnicas como IQR ou Z-score  
* Ajuda o modelo a entender melhor a relação entre variáveis  
* Um pouco menos subjetivo

3 \- Preenchimento de Valores Faltantes (NaN)

* Modelos de machine learning geralmente não funcionam com valores NaN  
* Muitas técnicas para preencher esses valores  
* Talvez seja melhor eliminar uma coluna com muitos valores NaN  
* Tomar cuidado que NaN pode representar apenas o valor zero  
* Razoavelmente Subjetivo

4  \- Codificação de Variáveis Categóricas

* Modelo de ML não aceitam strings para treinamento, apenas números  
* Podemos usar o get dummies, que cria uma coluna por categoria (*one-hot encoding*), mas pode gerar muitas colunas, se tiver vários valores únicos.  
* Podemos usar um ordinal encoder, que atribui às variáveis um valor único para cada uma, mas pode trazer uma noção falsa de ordem se mal utilizado.

5  \- Padronização dos dados

* Dados com escalas diferentes podem enganar o modelo  
* Técnicas como usar um Standard Scaler ajudam a padronizar os dados


  
