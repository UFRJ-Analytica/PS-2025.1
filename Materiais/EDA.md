# Análise exploratória dos dados (EDA)

## Tipos de EDA

Formalmente, existem quatro tipos de análises exploratórias. 

- **Univariado não gráfico.** Esta é a forma mais simples de análise de dados, onde os dados analisados consistem em apenas uma variável. Por ser uma variável única, não trata de causas ou relações. O principal objetivo da análise univariada é descrever os dados e encontrar padrões que existam neles.
- **Gráfico univariado.** Os métodos não gráficos não fornecem uma imagem completa dos dados. Métodos gráficos são, portanto, necessários. Os tipos comuns de gráficos univariados incluem:
Gráficos de caule e folhas, que mostram todos os valores dos dados e a forma da distribuição.
Histogramas, que são gráficos de barras em que cada barra representa a frequência (contagem) ou proporção (contagem/contagem total) de casos para um intervalo de valores.
Gráficos de caixa, que representam graficamente o resumo de cinco números de mínimo, primeiro quartil, mediana, terceiro quartil e máximo.
- **Não gráfico multivariado:** dados multivariados surgem de mais de uma variável. As técnicas multivariadas de EDA não gráfica geralmente mostram a relação entre duas ou mais variáveis dos dados por meio de tabulação cruzada ou estatística.
- **Gráfico multivariado:** os dados multivariados usam gráficos para exibir relacionamentos entre dois ou mais conjuntos de dados. O gráfico mais utilizado é um gráfico de barras agrupadas ou gráfico de barras, com cada grupo representando um nível de uma das variáveis e cada barra dentro de um grupo representando os níveis da outra variável.

> Mais detalhes sobre os tipos de gráficos na seção sobre a biblioteca `matplotlib`, que serve justamente para fazer gráficos

## Introdução
Uma análise exploratória de dados (abreviada por EDA - exploratory data analysis) é a principal maneira de  analisar e investigar conjuntos de dados e resumir suas principais características, muitas vezes empregando métodos de visualização de dados.

É um passo crucial quando se trabalha com dados, seja na construção de modelos ou para extrair qualquer tipo de informação dos mesmos, pois a partir da EDA que conseguimos estabelecer certezas sobre o que os dados representam, e a partir disso construir modelos ou análises mais profundas.

Muitas vezes, começamos a análise buscando respostas para uma pergunta, e acabamos encontrando outras dez perguntas sem respostas. Esse é exatamente o processo: em uma boa análise exploratória, o objetivo é aprender ao máximo com os dados, e isso frequentemente significa deixar de lado nossas ideias originais ou respostas pré-estabelecidas inconscientemente, e seguir apenas com aquilo que os dados nos mostra. 

Para fazer esse tipo de análise, podemos utilizar diversas técnicas, algumas extremamente simples e outras extremamente complexas. Abaixo, listamos algumas das consideradas essenciais para qualquer análise exploratória. Sinta-se livre para explorar outras fontes e descobrir novas maneiras de explorar seus dados.

## Ferramentas
Há uma infinidade de ferramentas que podem ser utilizadas nesse processo, mas aqui, vamos introduzir um pouco da linguagem python, que é bastante simples de ser usada e muito popular também (e portanto, há bastante material disponível online!).

As principais bibliotecas da linguagem que vamos abordar aqui são o `pandas` e o `matplotlib`. Encorajamos vocês a usarem o [jupyter notebook](https://jupyter.org/) ou o [google colab](https://colab.google.com/) para desenvolver os códigos, por facilitar a organização e visualização dos resultados.

### pandas

Pandas é a biblioteca que usamos para carregar e manipular os dados. Para usá-la no seu código python, precisamos importá-la:
```python
import pandas as pd
```
> OBS: `pd` é uma abreviação muito comum, usada até na documentação da linguagem, então vamos utilizá-la aqui também por consistência.

Ela define uma estrutura de dados chamada `DataFrame`, que é uma tabela: Cada linha é um registro, e cada coluna é um atributo. No dataset da tarefa, por exemplo, cada partida é uma linha, e cada coluna (Ex: Chutes a gol, Impedimentos) é um atributo.

Para carregar um `DataFrame` diretamente de um arquivo csv, como o dado na tarefa, o pandas tem uma função pronta:
```python
df = pd.read_csv("<caminho_do_arquivo>")
```

Ex: `df = pd.read_csv("Data/campeonatos_futebol_atualizacao.csv")`

Agora que temos o `DataFrame` carregado, vamos descobrir algumas informações sobre os dados

#### Resumindo dados

O pandas possui algumas funções que mostram uma versão resumida do seu dataframe, além de informações importante sobre o mesmo. Abaixo temos algumas das mais importantes:

`df.tail(x)` retorna um dataframe contendo os últimos *x* valores do seu conjunto de dados. Caso você omita o *x*, o valor padrão é 5.

`df.head(x)` retorna um dataframe com os primeiros *x* valores do seu conjunto de dados. Caso você omita o *x*, o valor padrão também é 5.

> Essas duas funções são especialmente boas quando se quer saber a “cara” de um conjunto de dados. 

`df.info()` Ajuda a obter uma visão geral rápida do conjunto de dados. Esta função retorna um breve resumo do dataframe, incluindo: Os tipos das colunas, número de valores não nulos e uso de memória.

`df.size()` retorna o número de elementos no seu dataframe.

`df.shape()` retorna o número de dimensões, bem como o tamanho de cada dimensão. Como os dataframes são bidimensionais, a forma que retorna é o número de linhas seguido do de colunas.

`df.describe()` Mostra uma tabela que é um resumo estatístico para colunas numéricas presentes no conjunto de dados. Mostra estatísticas como percentil, média e desvio padrão dos valores numéricos do DataFrame. Útil para entender a escala e a variação dos dados.

`df.sample(frac =.x)`  retorna uma amostra aleatória correspondente a  x% do dataframe. Ao usar `df.sample()` selecionamos apenas uma linha aleatória.

`df.nunique()` retorna o número de elementos únicos em cada coluna. Muito útil em características categóricas, especialmente nos casos em que não sabemos de antemão o número de categorias.

`df.columns()` retorna os nomes das colunas do seu dataframe em uma lista.

`df.nlargest(n,'x')`  Retorna um dataframe contendo as primeiras n linhas  da coluna chamada x, ordenadas por ordem decrescente.

`df.corr()`  Esta função é usada para encontrar a correlação de pares de todas as colunas no dataframe. Quaisquer valores ausentes são automaticamente excluídos. Qualquer coluna de tipo de dados não numérico é ignorada. Esta função é útil enquanto fazemos a seleção de quais colunas usar, observando a correlação entre os colunas e a variável de destino ou entre variáveis.

`df['x'].mean()` retorna o valor médio da coluna *x*. Podemos substitui *mean* por *median* ou *mode*.

`df['x'].min()` retorna o valor médio da coluna *x*. O análogo vale para o valor máximo.

`df['x'].quantile([.25, .5, .75])` retorna uma tabela contendo as estatísticas de quantil da coluna *x*, especificamente 50%, 25% e 75%.

#### Filtrando dados

Muitas vezes é necessário filtrar o dataframe, seja para valores em um intervalo específico, para valores exatos, entre outras possibilidades. Para isso existem infinitos métodos. Abaixo, através de exemplos, listamos alguns básicos e importantes de se saber.

No exemplo abaixo, por exemplo, substituímos  o dataframe *df_voos* por uma nova versão, onde os elementos da coluna *origem* são todos *Paris* e da coluna *tipo* são todos *Comercial*.  

```python
df_voos = df_voos[(df_voos.origem == "Paris") & (df_voos.tipo == "Comercial")]
```

O operador `~` é uitlizado para negar uma condição. Nesse cenário, caso o objetivo fosse um *dataframe* com todas as colunas, menos as colunas presentes no dataframe gerado acima, usaríamos o seguinte código:

```python
df_voos = df_voos[~((df_voos.origem == "Paris") & (df_voos.tipo == "Comercial"))]
```

 Também é possível realizar essa mesma operação usando uma linguagem semelhante a SQL.

```python
df_voos = df_voos.query('origem == "Paris" & tipo == "Comercial"')
```

```python
f_df = df.query('age > 25 and gender == "Male"')
```

Para usar variáveis nesse caso, é um pouco diferente do que estamnos acostumados com python:

```python
idade_busc = 25
genero_busc = 'Male'
f_df = df.query('idade > @idade_busc and gender == @genero_busc')
```

Os métodos mais utilizados para filtragem são `loc` e `iloc`  .

Podemos usar o loc para gerar o mesmo dataframe que do exemplo anterior.

```python
df_voos = df_voos.loc[(df_voos.origem == "Paris") & (df_voos.tipo == "Comercial")]
```

Já o `iloc`, usamos para selecionar linhas específicas por posição (digamos da segunda à quinta linha). Vale lembrar que a indexação em python começa do zero. `df.iloc[0:5,]` refere-se da primeira à quinta linha (excluindo o ponto final da 6ª linha aqui). `df.iloc[0:5,]` é equivalente a `df.iloc[:5,]`.

```python
df.iloc[:5,] #Primeiras 5 linhas
df.iloc[1:5,] #Segunda à Quinta linha
df.iloc[5,0] #Sexta linha e 1ª coluna
df.iloc[1:5,0] #Segunda à Quinta linha, primeira coluna
df.iloc[1:5,:5] #Segunda à quinta linha, primeiras 5 colunas
df.iloc[2:7,1:3] #Terceira a Sétima linha, 2ª e 3ª coluna
```

Também podemos filtrar pelo conteúdo específico de uma coluna.

Usando `.str`, você pode habilitar funções de string e aplicá-las no dataframe do pandas. `str[0]` significa primeira letra.

O exemplo abaixo seleciona linhas com valores começando pela letra 'A’ na coluna *var1.*

```python
df[df['var1'].str[0] == 'A']
```

A função `contains()` é semelhante à instrução LIKE em SQL. Você pode criar subconjuntos de dados mencionando o padrão na função `contains()`. O exemplo abaixo seleciona a string contendo as letras A ou B.

```python
df[df['var1'].str.contains('A|B')]
```

As vezes, ao criar colunas novas, precisamos renomeá-las. Para isso, usamos a função rename.

```python
df.rename(columns={'var1':'var1 renamed'}, inplace = True)
```

Para adicionar uma nova coluna no seu dataframe, existem diversas formas. Antes de tudo, certifique-se que a sua nova coluna tem a mesma quantidade de linhas que o seu dataframe. Sua nova coluna também deve ser do tipo `Series` para ser compatível com o seu dataframe.

```python
df['nova_coluna'] = pd.Series(dados_series, index=df1.index)
```

Para isso também podemos utilizar o `assign` .  No exemplo a seguir, criamos uma nova coluna *temp_f* no dataframe, que contém valores de temperatura convertidos de Celsius para Fahrenheit. Para isso usamos uma função lambda, uma função anônima em Python. Lambda é usada para definir uma função rápida e embutida sem definir formalmente uma função usando `def`.  `x.temp_c` acessa a coluna `temp_c` no dataframe e para cada valor dessa coluna, ele gera na mesma linha um novo valor na nova coluna `temp_f` , de acordo com a fórmula usada na função.

```python
df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)
```

As funções lambda são extremamente úteis quando sabemos usá-la. Para criar uma, usamos a seguinte notação:

```python
lambda arguments: expression
```

#### Limpeza dos dados

É comum que os dados recebidos sejam duplicados, rotulados incorretamente ou possuam valores faltantes. Se os dados estiverem incorretos ou inconsistentes, os resultados e algoritmos  fundados neles se tornam não confiáveis, mesmo que possam parecer corretos.  Por isso, realizamos a  limpeza e tratamento dos dados. O primeiro é o processo de corrigir ou remover dados incorretos, corrompidos, formatados incorretamente, duplicados ou incompletos dentro de um conjunto de dados.  O segundo refere-se ao conjunto de operações estratégicas aplicadas aos dados para torná-los úteis e relevantes para análise ou para um objetivo específico. Esses dois conceitos são separados por uma linha tênue,  mas o que importa é que ambos tem o mesmo objetivo: preparar os dados para seu objetivo final.

Não existe uma maneira absoluta ou passos exatos no processo de limpeza/tratamento de dados, pois os processos variam de um conjunto de dados para outro e do objetivo final. No entanto, é crucial estabelecer uma base para o seu processo de limpeza de dados.

Abaixo, vamos conhecer as principais etapas da limpeza e tratamento de dados. Todas elas possuem ao menos um código em python como exemplo. Para cada uma das etapas abaixo, existem infinitas maneiras de realiza-las em poucas ou muitas linhas de código, usando diferentes bibliotecas e abordagens diferentes para casos mais específicos.

**Dados duplicados**: Para verificar se o seu DataFrame possui valores duplicados em Python usando pandas, você pode usar o método `duplicated()`. Este método retornará uma série booleana indicando se cada linha é duplicada ou não.
```python
df = pd.DataFrame(data) #Definindo dataframe
df.duplicated() # Retorna um array de booleanos (True/False)
```

Para simplificar a saída, você pode utilizar a linha de código abaixo, cuja a saída é apenas a  posição das linha duplicadas, ao invés de uma série. 
```python
df[df.duplicated()].index
```

Em alguns casos, não queremos eliminar duplicatas exatas, mas sim linhas que possuem valores duplicados em apenas uma coluna. Por exemplo, caso seus dados possuam uma coluna com o código de usuário do usuário de um sistema, e uma segunda coluna com a data de nascimento do usuário, podemos querer confirmar que um código não esteja duplicado, pois o mesmo não pode estar associado a duas datas de nascimento diferentes. Nesse caso, basta usar o método duplicated em apenas uma coluna, ao invés de em todo o dataframe.

```python
df["UserId"].duplicated()
```

**Dados Inconsistentes**:Nessa etapa, buscamos verificar e ajustar valores que estão inconsistentes em termos de formato ou padrão. Por exemplo, na forma de escrita de dados categóricos, uma letra maiúscula ou minúscula ou uma palavra com ou sem acento, criam uma categoria completamente diferente no seu conjunto de dados. 

No exemplo abaixo, vamos substituir todos os valores da coluna `Nome` de um dataframe, pelos mesmos valores, porém substituindo todas as letras maiúsculas por minúsculas.

```python
df['Nome'] = df['Nome'].str.lower()
```

Dessa forma, se futuramente procurarmos por pessoas com o mesmo nome, "gabriel", "Gabriel" e "GABRIEL" serão considerados iguais.

**Dados faltantes**: Para identificar dados faltosos, podemos usar `df.isna()` ou `df.isnull()` , ambos retornam um dataframe contendo valores booleanos, sendo que *True* representa um valor faltante.


```python
df.isna().sum() #Retorna quantidade de valores vazios por coluna.

df = df.dropna() # Remove valores vazios do conjunto

df = df.dropna(axis=1) # Remove colunas com valores vazios

df = df.dropna(axis=0) # Remove linhas com valores vazios

df = df.dropna(subset=['A']) #Remove linhas com valores vazios na coluna 'A'
```

Nem sempre é boa ideia excluir certas linhas ou colunas do seu conjunto de dados. Por isso, também é recomendado substituir os valores vazios. Para isso existem diversas estratégias.

```python
df = df.fillna(value={'A': 0, 'B': 'missing'}) #Substitui valores vazios por 0 na coluna 'A' e por 'missing', na coluna 'B'.

df = df.interpolate() #Subistitui valores vazios por estimitivas que consideram todo o dataset, usando métodos de interpolação numérica

df['A'].fillna(df['A'].mean())  #Substitui pela média da coluna (Para colunas numéricas)

df['B'].fillna(df['B'].mode()[0]) #Substitui pela moda da coluna (Para colunas categóricas)
```

A estratégia correta depende muito do contexto e do significado dos dados. Claro que em muitos casos, a melhor estratégia pode ser de fato apagar a coluna (Ex: Se quase todos os registros são faltantes, e portanto não conseguimos comparar uma linha com outra com base nessa coluna)

**Ruído**: Para ter um conjunto de dados mais representativo e confiável, as vezes é necessário remover outliers e ruído.

Por exemplo, suponha que você possui um conjunto de dados com uma amostra aleatória de 100 pessoas da sua cidade. Sua cidade é muito humilde, mas existem 3 bilionários que moram lá, e levam um estilo de vida muito diferente das demais pessoas. Por acaso, os 3 bilionários estão presentes no seu conjunto de dados e mantê-los lá atrapalharia qualquer análise sobre o perfil econômico da população que você pudesse conduzir, logo, o ideal é remove-los do dataset. Para isso, podemos filtra-lo.

```python
df = df[(df['networth'] < 1000000000)]
```

#### Tratamento dos dados
**Tipagem e dados inválidos**: Muitas vezes, é necessário identificar e corrigir dados que foram corrompidos ou que não são válidos para o contexto. Um exemplo é um campo que deveria conter números, mas pode ter letras. Esses registros devem ser corrigidos ou removidos, para que a coluna possua apenas um único tipo de dado.

```python
df.dtypes # Retorna o tipo de dado de todas as colunas (Caso esteja padronizada)

df.B.dtype #Retorna o tipo de dado da coluna B. (Caso esteja padronizada)

df['A'] = df['A'].astype(float) #Converte todos os valores da coluna 'A' para float

df['A'] = pd.to_numeric(df['A']) #Converte para int ou float, dependendo dos dados

df = df.convert_dtypes() #Altera todas as colunas para os tipos apropriados
```

**Agregação**: Em alguns casos, é necessário resumir ou agrupar seus dados para criar estatísticas ou métricas mais gerenciáveis. Por exemplo,  agrupar dados de transações diárias para calcular a receita mensal total.

Em Python, você pode usar o método aggregate() para realizar várias operações de agregação em um DataFrame do pandas. O método permite aplicar uma função ou uma lista de nomes de funções a serem executadas ao longo de um dos eixos do DataFrame, padrão 0, que é o eixo das linhas/ índices.

As agregações usadas com mais frequência são:

- `sum`: Retorna a soma dos valores do eixo solicitado
- `min`: Retorna o mínimo dos valores para o eixo solicitado
- `max`: Retorna o máximo dos valores para o eixo solicitado
- `count` : Retorna o número total de itens.

Assim, podemos usar o método `aggregate()` de duas formas:

```python
df.aggregate(['sum', 'min'])
```
Para exibir - nesse caso - a soma e o mínimo de todas as colunas numéricas disponíveis. Ou

```python
df.aggregate({"Number":['sum', 'min'], 
              "Age":['max', 'min'], 
              "Weight":['min', 'sum'],  
              "Salary":['sum']})
```

Para exibir os valores das agregações especificadas para cada coluna também especificada.

### matplotlib
matplotlib é a biblioteca mais usada em python para desenhar gráficos de datasets. Nesta seção, vamos dar exemplos de uso de algumas funções do matplotlib, além de uma descrição de cada um dos respectivos tipos de gráficos. Se você não conhece as funções mostradas, recomendamos que execute os códigos para ver os resultados.

Antes de seguir com os exemplos, vale dar algumas definições quanto às categorias de dados:

**Dados Quantitativos* são dados numéricos, que podem ser medidos em alguma grandeza. Eles podem ser contínuos (assumindo qualquer valor real), discretos (assumindo um determinado conjunto de valores pré-estabelecido) ou binários (assumindo apenas os valores 0 ou 1, ou verdadeiro ou falso).

**Dados Qualitativos** são dados descritivos. Normalmente são usados para categorizar ou classificar algo; subdividir o dataset. Podem ser divididos em dados nominais (categorias sem uma ordem intrínseca, como cores) ou ordinais (quando há uma ordem implícita, como em tamanhos de camisa)

#### Gráficos de barras
Gráficos de barras são ideais para comparar categorias diferentes de dados qualitativos ou quantitativos discretos. Nele, cada barra representa uma categoria (ou valor numérico discreto), e sua altura ou comprimento é proporcional à frequência de ocorrência ou valor agregado (ex: Média por categoria). Pode ser horizontal ou vertical. O modelo vertical é recomendado quando há poucas categorias. 

```python
import numpy as np
import matplotlib.pyplot as plt 

#Definindo DataFrame
data = {'C':20, 'C++':15, 'Java':30, 
        'Python':35} #Criamos um diconário
courses = list(data.keys())
values = list(data.values()) #Transformamos as chaves e valores associados do dicionário em duas listas
 
fig = plt.figure(figsize = (10, 5)) #Definindo Tamanho da Imagem
plt.bar(courses, values, color ='maroon', 
        width = 0.4) #Configurando gráfico de barras

#Definindo legendas e plotando
plt.xlabel("Cursos Oferecidos")
plt.ylabel("No. Estudantes")
plt.title("Estudantes por Curso Oferecido")
plt.show()
```

#### Histograma
Histogramas são usado para mostrar a distribuição de variáveis, traçar dados quantitativos e identificar a proporção dos dados que ocorre dentro de determinado intervalo de valores. Os histogramas possuem dados quantitativos em ambos os eixos, em vez de relacionar informações sobre categoria, como no gráfico de barras. Cada barra do histograma é desenhado a partir da base do gráfico (eixo X) até uma altura correspondente à proporção dos dados que está nesse intervalo específico. A largura de cada barra é proporcional à largura do intervalo de valores que ela representa.

É comum utilizar escalas diferentes para os eixos vertical e horizontal nesse tipo de gráfico

```python
import matplotlib.pyplot as plt
import numpy as np

# Generando dados aleatórios
data = np.random.randn(1000)

# Plotando histograma básico
plt.hist(data, bins=10, color='maroon', edgecolor='black') # a qtd de bins é o número de colunas no histograma

# Adicionando legendas e título
plt.xlabel('Valores')
plt.ylabel('Frequências')
plt.title('Histograma')

#Exibindo gráfico
plt.show()
```
> Nesse gráfico podemos visualizar a distribuição de uma variável pela frequência dos valores assumidos por ela.  Podemos ver, por exemplo, que valores próximos do intervalo entre 0 e 0.5 aparecem cerca de 250 vezes no nosso conjunto de dados aleatórios.

#### Gráfico de Linhas
Um gráfico de linhas é uma representação visual de dados em que pontos de dados são marcados em um plano cartesiano e conectados por linhas. Cada ponto representa um valor no eixo y em função de outro valor no eixo x. Ele mostra as variações de uma variável de acordo com outra variável.

```python
import matplotlib.pyplot as plt 

#Definindo Dataset
days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
temperature = [36.6, 37, 37.7,39,40.1,43,43.4,45,45.6,40.1,44,45,46.8,47,47.8]

#Defininfo plot
plt.plot(days, temperature)

#Definindo título e legendas
plt.title("Temperatura em Dehli")
plt.xlabel("Dias")
plt.ylabel("Temperatura")
plt.show()
```
> Nesse gráfico podemos visualizar a temperatura na cidade de Dehli no decorrer de 15 dias.

#### Gráfico de Pizza
Um gráfico de pizza, também conhecido como gráfico de setores ou gráfico circular, é uma representação visual usada para ilustrar a composição de um todo através de partes proporcionais. Cada "fatia" ou "setor" do gráfico representa uma categoria e seu tamanho é proporcional à quantidade ou percentual dessa categoria em relação ao total. 

```python
# Importando bibliotecas
from matplotlib import pyplot as plt
import numpy as np

# Criando dataset
cars = ['AUDI', 'BMW', 'FORD',
        'TESLA', 'JAGUAR', 'MERCEDES']

stock = [23, 17, 35, 29, 12, 41]

# Configurando
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=cars, autopct='%1.1f%%')
plt.title('Porcentagem de Carros no Estoque')

# Mostrando plot
plt.show()
```
> Nesse gráfico podemos ver a composição de um estoque composto por diferentes tipos de carros.

#### Gráfico de Dispersão (Scatter Plot)
Um **scatter plot** (ou gráfico de dispersão) é um tipo de gráfico utilizado para visualizar a relação entre duas variáveis diferentes. Cada ponto no gráfico representa uma observação com coordenadas correspondentes aos seus respectivas valores nas duas variáveis. É uma ferramenta útil para identificar padrões, correlações ou tendências em um conjunto de dados.

Se os pontos formarem um padrão que se eleva da esquerda para a direita, por exemplo, isso indica que conforme uma variável aumenta, a outra também tende a aumentar; logo, existe uma correlação positiva. Se os pontos estiverem distribuídos aleatoriamente, sem formar um padrão claro, isso indica que não há uma correlação linear entre as variáveis.

```python
import matplotlib.pyplot as plt

x =[5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6] 

y =[99, 86, 87, 88, 100, 86, 103, 87, 94, 78, 77, 85, 86]

plt.scatter(x, y, c ="maroon")
plt.show()
```
> Nesse gráfico podemos ver que as variáveis x e y possuem uma correlação negativa considerável: Conforme x aumenta, y diminui.

#### Gráfico de Caixa (Box Plot)

O Box Plot é particularmente útil para visualizar a mediana, a dispersão, os outliers e a assimetria dos dados. Essas métricas mostram a distribuição com base em um resumo estatístico. Ele possui os seguintes componentes:

1. **Caixa (Box):** A caixa central do gráfico representa o intervalo interquartil (IQR), que contém 50% dos dados. A parte inferior da caixa é o primeiro quartil (Q1), e a parte superior é o terceiro quartil (Q3).
2. **Linha Central (Mediana):** Dentro da caixa, uma linha marca a mediana (Q2) dos dados, que é o ponto onde metade dos dados está abaixo e metade está acima.
3. **Extensões (Whiskers):** As "extensões" ou "bigodes" se estendem das extremidades da caixa até os menores e maiores valores dos dados que não são considerados outliers. Esses limites são geralmente definidos como 1,5 vezes o IQR a partir dos quartis (Q1 - 1,5*IQR e Q3 + 1,5 IQR*).
4. **Outliers:** Valores que caem fora do intervalo dos bigodes são considerados outliers e são representados por pontos ou asteriscos.


```python
# Importando bibliotecas
import matplotlib.pyplot as plt
import numpy as np

#Gerando dataset aleatório
np.random.seed(10)

data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data_1, data_2, data_3, data_4]

# configurando plot
fig = plt.figure(figsize =(10, 7))
ax = fig.add_axes([0, 0, 1, 1])

# Gerando plot
bp = ax.boxplot(data)
plt.show()
```
> Nesse gráfico, podemos ver o box plot de quatro conjuntos de dados diferentes. Podemos ver que todos os conjuntos possuem uma distribuição simétrica, apesar da dispersão aumentar gradativamente entre eles.

| Tipo de Gráfico | Uso Ideal |  |
| --- | --- | --- |
| de Barras | comparar categorias diferentes de dados qualitativos e quantitativos discretos | Comparação |
| Histograma | representar a  distribuição de dados contínuos ou discretos.  | Distribuição |
| de Linhas | mostrar tendências em dados contínuos ao longo do tempo | Comparação |
| Pizza | examinar relações parte-todo simples. | Composição |
| de Dispersão (Scatter) | mostrar como os dados são distribuídos com duas variáveis, especialmente bom para grandes conjuntos de dados. | Distribuição |
| de Caixa (Box Plot) | mostrar como os dados são distribuídos com base em um resumo estatístico de cinco números. Uma pequena “caixa” indica que a maioria dos dados está dentro de um intervalo consistente, enquanto uma caixa maior indica que os dados estão distribuídos de forma mais ampla. | Distribuição |

## Material complementar

[Artigo da IBM sobre análise exploratória](http://ibm.com/think/topics/exploratory-data-analysis)

[Documentação do matplotlib](https://matplotlib.org/), que inclui tutoriais

[Documentação do pandas](https://pandas.pydata.org/), que também inclui muitos exemplos