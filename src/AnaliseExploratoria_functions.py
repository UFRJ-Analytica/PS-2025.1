import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy.stats import probplot
from sklearn.impute import SimpleImputer, KNNImputer

def calcular_resultado(df):
    """
    Calcula o resultado do confronto com base nas colunas 'Gols 1' e 'Gols 2'.
    - O DataFrame original não é modificado.
    - O resultado é retornado em uma pandas Series com o nome "resultados".

    Parâmetros:
    df (pandas.DataFrame): DataFrame que contém as colunas 'Gols 1' e 'Gols 2'.

    Retorna:
    pandas.Series: Série com os resultados, onde:
        - "Vitória" se 'Gols 1' > 'Gols 2'
        - "Empate" se 'Gols 1' == 'Gols 2'
        - "Derrota" se 'Gols 1' < 'Gols 2'
    """
    resultados = []
    for _, row in df.iterrows():
        if row['Gols 1'] > row['Gols 2']:
            resultados.append("Vitória")
        elif row['Gols 1'] == row['Gols 2']:
            resultados.append("Empate")
        else:
            resultados.append("Derrota")
    return pd.Series(resultados, name="resultado")

def substituir_tiros_por_chutes(df):
    """
    Recebe um DataFrame e retorna uma nova cópia onde:
      - Em 'Tiro de meta 2', se houver NaN, o valor é substituído pelo de 'Chutes fora 1'.
      - Em 'Tiro de meta 1', se houver NaN, o valor é substituído pelo de 'Chutes fora 2'.

    O DataFrame original não é modificado.
    """

    df_copy = df.copy()

    # Criação de máscara para linhas onde 'Tiros de Meta 2' é NaN
    mask_tiros_meta2 = df_copy['Tiro de meta 2'].isnull()
    # Substitui pelos valores correspondentes da coluna 'Chute Fora 1'
    df_copy.loc[mask_tiros_meta2, 'Tiro de meta 2'] = df_copy.loc[mask_tiros_meta2, 'Chutes fora 1']

    # Análogo para 'Tiros de Meta 1'
    mask_tiros_meta1 = df_copy['Tiro de meta 1'].isnull()
    df_copy.loc[mask_tiros_meta1, 'Tiro de meta 1'] = df_copy.loc[mask_tiros_meta1, 'Chutes fora 2']

    return df_copy

from typing import Tuple
def histogram_and_stats(df: pd.DataFrame, column: str, plot: bool = True, interactive: bool = True, bins: int = 40)\
        -> Tuple[int, int, float, float, float, float, float, float, float, float, float]:
    """
    Plota (opcionalmente) um histograma para a coluna especificada de um DataFrame e retorna e printa estatísticas descritivas.

    :param df: O DataFrame que contém os dados.
    :type df: pd.DataFrame
    :param column: O nome da coluna para a qual será gerado o histograma.
    :type column: str
    :param plot: Se True, gera o gráfico. Se False, apenas retorna as estatísticas.
    :type plot: bool
    :param interactive: Se True, o plot é feito pelo plotly (interativo. Se False, pelo matplotib.
    :type interactive: bool
    :param bins: Ajuste para o gráfico do histograma alinhar corretamente.
    :type bins: int


    :return: Uma tupla contendo:
        - num_nan (int): Número de valores NaN
        - num_valid (int): Número de valores não NaN
        - media (float): Média dos valores (excluindo NaN)
        - desvio (float): Desvio padrão dos valores (excluindo NaN)
        - minimo (float): Valor mínimo (excluindo NaN)
        - maximo (float): Valor máximo (excluindo NaN)
        - percentil_5 (float): 5º percentil
        - percentil_25 (float): 25º percentil (1º quartil)
        - percentil_50 (float): 50º percentil (mediana)
        - percentil_75 (float): 75º percentil (3º quartil)
        - percentil_95 (float): 95º percentil
    :rtype: Tuple[int, int, float, float, float, float, float, float, float, float, float]

    Exemplo de uso:
        >>> stats = histogram_and_stats(df, 'Chutes a gol 1')
        >>> print("Estatísticas:", stats)
    """

    # Seleciona a coluna e remove os NaN para o histograma
    data = df[column]
    data_sem_nan = data.dropna()

    # Cálculo das estatísticas
    num_nan = data.isna().sum()
    num_valid = data.notna().sum()
    media = data_sem_nan.mean()
    desvio = data_sem_nan.std()
    minimo = data_sem_nan.min()
    maximo = data_sem_nan.max()
    percentil_5 = data_sem_nan.quantile(0.05)
    percentil_25 = data_sem_nan.quantile(0.25)
    percentil_50 = data_sem_nan.quantile(0.50)
    percentil_75 = data_sem_nan.quantile(0.75)
    percentil_95 = data_sem_nan.quantile(0.95)

    if plot == True:
        if not interactive:
            plt.figure(figsize=(6, 4))
            plt.hist(data_sem_nan, bins=bins, edgecolor='black', histtype='stepfilled')
            plt.title(f'{column}')
            plt.xlabel(column)
            plt.ylabel('Ocorrências')
            plt.grid(False)
            plt.axvline(media, linewidth=1, label='Média')
            plt.axvline(percentil_50, linewidth=1, label='Mediana')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            fig = px.histogram(
                x=data_sem_nan,
                #nbins='auto',
                title=f'{column}',
                labels={"x": column, "y": "Ocorrências"}
            )

            # Adiciona linhas verticais para a média e a mediana
            fig.add_vline(
                x=media,
                line_width=1,
                line_dash="dash",
                line_color="red",
                annotation_text="Média",
                annotation_position="top right"
            )
            fig.add_vline(
                x=percentil_50,
                line_width=1,
                line_dash="dash",
                line_color="green",
                annotation_text="Mediana",
                annotation_position="top right"
            )
            fig.show()

    print(
        f"\033[1mNaN's:\033[0m {num_nan} --|-- \033[1mOcorrências:\033[0m {num_valid} --|-- \033[1mMédia:\033[0m {media:.2f} --|-- "
        f"\033[1mDesv. padrão:\033[0m {desvio:.2f} --|-- \033[1mMínimo:\033[0m {minimo:.2f} --|-- \033[1mMáximo:\033[0m {maximo:.2f} --|-- "
        f"\033[1m5º percentil:\033[0m {percentil_5:.2f} --|-- \033[1m25º percentil:\033[0m {percentil_25:.2f} --|-- "
        f"\033[1m50º percentil (mediana):\033[0m {percentil_50:.2f} --|-- \033[1m75º percentil:\033[0m {percentil_75:.2f} --|-- "
        f"\033[1m95º percentil:\033[0m {percentil_95:.2f}\n--------------")

    return (int(num_nan),int(num_valid),float(media),float(desvio),int(minimo),int(maximo),float(percentil_5),float(percentil_25),float(percentil_50),float(percentil_75),float(percentil_95))


def evaluate_distribution(df: pd.DataFrame, column: str, show_boxplot: bool = False,bins: int = 40) -> dict:
    """
    Avalia a distribuição de uma coluna do DataFrame calculando skewness e
    (excesso de) kurtosis e gerando gráficos para visualização: histograma com KDE,
    Q-Q plot e opcionalmente um boxplot.

    Parameters:
        df (pd.DataFrame): DataFrame contendo os dados.
        column (str): Nome da coluna a ser avaliada.
        show_boxplot (bool, optional): Se True, inclui o boxplot na visualização.
                                    Padrão é False.
        bins (int, optional): Número de intervalos no histograma. O default é 40, mas pode ser alterado para
                    uma melhor visualização

    Returns:
        dict: Dicionário com os seguintes valores:
            - "skewness": Valor da assimetria da distribuição.
            - "kurtosis": Valor do excesso de kurtosis (kurtosis - 3).
    """
    # Remove os NaN para os cálculos e plots
    dados = df[column].dropna()

    skewness_value = dados.skew()
    kurtosis_value = dados.kurt()

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(dados, kde=True,bins=bins)
    plt.title(f'Histograma de {column}\nSkew: {skewness_value:.2f} | Kurtosis (excesso): {kurtosis_value:.2f}')

    # Q-Q Plot
    plt.subplot(1, 3, 2)
    probplot(dados, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot de {column}')

    plt.subplot(1, 3, 3)
    if show_boxplot:
        sns.boxplot(x=dados)
        plt.title(f'Boxplot de {column}')
    else:
        plt.axis('off')

    plt.show()

    print(f"\nColuna: {column}")
    print(f"Skewness: {skewness_value:.3f}")
    print(f"Kurtosis (excesso): {kurtosis_value:.3f}")
    return {
        "skewness": skewness_value,
        "kurtosis": kurtosis_value
    }

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

def impute_missing_values(df: pd.DataFrame, columns: list or str, strategy: str = "median") -> pd.DataFrame:
    """
    Imputa os valores faltantes (NaN) nas colunas especificadas do DataFrame utilizando
    a estratégia definida. As opções de estratégia são:

    - "mean": substitui os NaN pela média dos valores.
    - "median": substitui os NaN pela mediana dos valores.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados.
        columns (list or str): Lista de nomes das colunas onde os valores faltantes serão imputados,
                            ou uma string para uma única coluna.
        strategy (str, opcional): Estratégia de imputação ("mean" ou "median").
                                Padrão é "median".

    Retorna:
        pd.DataFrame: DataFrame atualizado com os valores faltantes imputados.

    Raises:
        ValueError: Se a estratégia fornecida não for "mean", "median" ou "knn".
    """
    if strategy not in ["mean", "median"]:
        raise ValueError("A estratégia deve ser 'mean' ou 'median'.")

    # Se 'columns' for uma string, converte para lista
    if isinstance(columns, str):
        columns = [columns]

    # Usamos .loc para evitar o SettingWithCopyWarning
    if strategy in ["mean", "median"]:
        imputer = SimpleImputer(strategy=strategy)
        df.loc[:, columns] = imputer.fit_transform(df.loc[:, columns])

    return df

from sklearn.preprocessing import StandardScaler
from typing import Union, List
def standardize_and_knn_impute(
        df: pd.DataFrame,
        columns: Union[List[str], str],
        n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Imputa primeiro usando KNN (na escala original) e depois padroniza (z-score).

    Parameters:
        df (pd.DataFrame): DataFrame com colunas a serem processadas.
        columns (list[str] | str): Nome(s) das colunas.
        n_neighbors (int): Número de vizinhos do KNNImputer.

    Returns:
        pd.DataFrame: DataFrame com imputação e padronização.
    """
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler

    if isinstance(columns, str):
        columns = [columns]

    df = df.copy()

    # Passo 1: Imputa com KNN sobre dados originais
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_vals = imputer.fit_transform(df[columns])

    # Garante que não há valores negativos pós-imputação
    imputed_vals[imputed_vals < 0] = 0

    # Atualiza o DataFrame
    df[columns] = imputed_vals

    # Passo 2 (opcional, só se precisar para o modelo):
    # Padroniza os valores já imputados (se o modelo exigir)
    scaler = StandardScaler()
    standardized_vals = scaler.fit_transform(df[columns])

    # Pode armazenar as variáveis padronizadas com outro nome, se preferir:
    for i, col in enumerate(columns):
        df[f'{col}_padronizada'] = standardized_vals[:, i]

    return df



def position_to_binary(
        df: pd.DataFrame,
        columns: List[str],
        threshold_backline: int = 3,
        threshold_strikers: int = 3,
        replace: bool = False) -> pd.DataFrame:
    """
    Converte formações táticas em indicador binário:

      - 1 (ofensiva) se:
          * número de defensores ≤ threshold_backline,  OR
          * número de atacantes  ≥ threshold_strikers
      - 0 (defensiva) caso contrário
      - NaN de entrada permanece NaN

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original com colunas de formação (e.g. '4-2-3-1').
    columns : List[str]
        Nome(s) da(s) coluna(s) a converter, ex.: ['Position 1', 'Position 2'].
    threshold_backline : int, default=3
        Máx. de zagueiros para classificar logo como ofensiva.
    threshold_strikers : int, default=3
        Mín. de atacantes para classificar como ofensiva.
    replace : bool, default=False
        - True → sobrescreve a(s) coluna(s) original(is) com 0/1/NaN.
        - False → mantém originais e cria novas com sufixo '_bin'.

    Returns
    -------
    pd.DataFrame
        Cópia de `df` com as colunas transformadas.
    """
    out = df.copy()

    for col in columns:
        # extrai defensores (primeiro bloco) e atacantes (último bloco)
        backs = pd.to_numeric(
            out[col].str.split('-', n=1, expand=True)[0],
            errors='coerce'
        )
        strikers = pd.to_numeric(
            out[col].str.rsplit('-', n=1, expand=True)[1],
            errors='coerce'
        )

        # critério composto
        mask_off = (backs <= threshold_backline) | (strikers >= threshold_strikers)

        # montar série float pra aceitar NaN
        binary = mask_off.astype(float)
        # preserva NaN onde a formação original faltava ou parse falhou
        binary[out[col].isna() | backs.isna() | strikers.isna()] = np.nan

        # atribuir resultado
        if replace:
            out[col] = binary
        else:
            out[f"{col}_bin"] = binary
    return out
from typing import Union, List, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def normalizar_variaveis(
        df: pd.DataFrame,
        columns: Union[List[str], str],
        method: Literal['zscore', 'minmax', 'robust'] = 'zscore'
) -> pd.DataFrame:
    """
    Normaliza as variáveis especificadas usando diferentes técnicas:

    Methods:
        - 'zscore': padronização (média 0, desvio 1).
        - 'minmax': normalização para o intervalo [0,1].
        - 'robust': normalização baseada em mediana e IQR (menos sensível a outliers).

    Parameters:
        df (pd.DataFrame): DataFrame contendo as colunas a normalizar.
        columns (list[str] | str): Nome ou lista de nomes das colunas.
        method: Técnica de normalização a aplicar.

    Returns:
        pd.DataFrame: DataFrame com as colunas informadas normalizadas.
    """
    if isinstance(columns, str):
        columns = [columns]
    df = df.copy()

    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Método de normalização '{method}' não reconhecido.")

    df.loc[:, columns] = scaler.fit_transform(df.loc[:, columns])
    return df

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def criar_indices(
        df: pd.DataFrame,
        norm_method: Literal['zscore', 'minmax', 'robust'] = 'minmax'
) -> pd.DataFrame:
    """
    Normaliza variáveis originais, calcula índices defensivos e ofensivos,
    preenche colunas faltantes com zero, evita KeyError e garante índices não-negativos.

    - Usa minmax por padrão para evitar valores negativos.
    - Colunas originais ausentes viram zero (nenhuma ação registrada).
    - Índices clamped para >= 0.

    Parâmetros:
      df: DataFrame com variáveis originais e 'Position 1/2'.
      norm_method: técnica de normalização.

    Retorna DataFrame somente com índices derivados.
    """
    df = df.copy()
    eps = 1e-5

    orig_cols = [
        'Defesas difíceis 1', 'Defesas difíceis 2',
        'Chutes bloqueados 1', 'Chutes bloqueados 2',
        'Tiro de meta 1', 'Tiro de meta 2',
        'Contra-ataques 1', 'Contra-ataques 2',
        'Faltas 1', 'Faltas 2',
        'Cartões amarelos 1', 'Cartões amarelos 2',
        'Cartões vermelhos 1', 'Cartões vermelhos 2',
        'Chutes a gol 1', 'Chutes a gol 2',
        'Chutes fora 1', 'Chutes fora 2',
        'Escanteios 1', 'Escanteios 2',
        'Impedimentos 1', 'Impedimentos 2',
        'Gols 1', 'Gols 2',
        'Posse 1(%)', 'Posse 2(%)',
        'Position 1', 'Position 2'
    ]

    # preencher faltantes para evitar KeyError
    for col in orig_cols:
        if col not in df.columns:
            df[col] = 0

    # normaliza apenas o que existe
    df = normalizar_variaveis(df, orig_cols, method=norm_method)

    # fatores táticos
    def1 = (1 - df['Position 1'])
    def2 = (1 - df['Position 2'])
    ofs1 = df['Position 1']
    ofs2 = df['Position 2']

    # defensivos
    df['Desgaste_Defensivo_1'] = (
                                         3*df['Defesas difíceis 1'] + df['Tiro de meta 1'] + df['Faltas 1'] +
                                         2*df['Cartões amarelos 1'] + 5*df['Cartões vermelhos 1']
                                 ) * def1
    df['Desgaste_Defensivo_2'] = (
                                         3*df['Defesas difíceis 2'] + df['Tiro de meta 2'] + df['Faltas 2'] +
                                         2*df['Cartões amarelos 2'] + 5*df['Cartões vermelhos 2']
                                 ) * def2

    df['Barreira_Defensiva_1'] = (
                                         4*df['Defesas difíceis 1'] + 3*df['Chutes bloqueados 1'] +
                                         2*df['Contra-ataques 2'] + df['Impedimentos 2']
                                 ) * def1
    df['Barreira_Defensiva_2'] = (
                                         4*df['Defesas difíceis 2'] + 3*df['Chutes bloqueados 2'] +
                                         2*df['Contra-ataques 1'] + df['Impedimentos 1']
                                 ) * def2

    # ofensivos
    df['Pressao_Ofensiva_1'] = (
                                       2*df['Chutes a gol 1'] + df['Chutes fora 1'] +
                                       df['Escanteios 1'] + 1.5*df['Impedimentos 1']
                               ) * ofs1
    df['Pressao_Ofensiva_2'] = (
                                       2*df['Chutes a gol 2'] + df['Chutes fora 2'] +
                                       df['Escanteios 2'] + 1.5*df['Impedimentos 2']
                               ) * ofs2

    df['Eficiencia_Finalizacao_1'] = (df['Gols 1'] / (df['Chutes a gol 1'] + df['Chutes fora 1'] + eps)) * ofs1
    df['Eficiencia_Finalizacao_2'] = (df['Gols 2'] / (df['Chutes a gol 2'] + df['Chutes fora 2'] + eps)) * ofs2

    df['Variedade_Ofensiva_1'] = (
                                         df['Chutes a gol 1'] + df['Chutes fora 1'] +
                                         df['Escanteios 1'] + df['Impedimentos 1']
                                 ) * ofs1
    df['Variedade_Ofensiva_2'] = (
                                         df['Chutes a gol 2'] + df['Chutes fora 2'] +
                                         df['Escanteios 2'] + df['Impedimentos 2']
                                 ) * ofs2

    df['Controle_Jogo_1'] = df['Posse 1(%)'] * ofs1
    df['Controle_Jogo_2'] = df['Posse 2(%)'] * ofs2

    # garante não-negatividade dos índices
    idx_cols = [c for c in df.columns if any(key in c for key in ['Desgaste', 'Barreira', 'Pressao', 'Eficiencia', 'Variedade', 'Controle'])]
    df[idx_cols] = df[idx_cols].clip(lower=0)

    # remove originais
    df.drop(columns=orig_cols, inplace=True)
    return df