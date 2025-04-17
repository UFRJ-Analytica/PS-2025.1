import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Função que retorna um dataframe contendo os valores faltantes de cada coluna
def na_percent(df):
    sorted_sum = df.isna().sum().sort_values(ascending=True) / len(df)
    sorted_sum = sorted_sum[sorted_sum > 0]
    return sorted_sum

# Função que retorna um barplot já estilizado com a porcentagem de colunas faltando
def missplot(df):
    missing_values = na_percent(df)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    sns.barplot(y=missing_values.index, x=missing_values.values, hue=missing_values.index, palette="Greys", ax=ax, orient="h")

    ax.set_xticks([i / 10 for i in range(0, 11)])
    ax.tick_params(axis='y', labelsize=5)
    ax.tick_params(axis='x', labelsize=6)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_ylabel("")
    ax.set_xlabel("Porcentagem de Nulos", fontsize=6)
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_visible(False)

    fig.suptitle('Porcentagem de Valores Faltantes entre as Colunas', x=0.048, ha="left", fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


#Função para remover outliers
def removedor_outliers(df, col, fator=1.5):
    q25, q75 = np.percentile(df[col], [25, 75])
    iqr = q75 - q25
    limite_inferior = q25 - fator * iqr
    limite_superior = q75 + fator * iqr
    df_filtrado = df[(df[col] >= limite_inferior) & (df[col] <= limite_superior)]
    print(f"{col}: {len(df) - len(df_filtrado)} outliers removidos.")
    return df_filtrado

#Função que define algoritmo para preenchimento de NaNs
def fillna_logic(df, stat):
    col1_name = stat + ' 1'
    col2_name = stat + ' 2'
    col1 = df[col1_name]
    col2 = df[col2_name]

    if col1.dtype == 'object' or col2.dtype == 'object':
        moda1 = col1.mode().iloc[0]
        moda2 = col2.mode().iloc[0]
        df[col1_name] = col1.fillna(moda1)
        df[col2_name] = col2.fillna(moda2)

    else:
        stat_mean = (col1 + col2).mean()
        stat_diff_mean = abs(col1 - col2).mean()
        nan_mask = col1.isna() | col2.isna()

        for idx in nan_mask[nan_mask].index:
            base = stat_mean / 2
            noise = np.random.randint(-stat_diff_mean, stat_diff_mean)

            if np.random.rand() > 0.5:
                fill1 = np.floor(base) + noise
            else:
                fill1 = np.ceil(base) + noise

            if pd.isna(col1.loc[idx]):
                df.at[idx, col1_name] = fill1
                df.at[idx, col2_name] = stat_mean - fill1

    return df[col1_name], df[col2_name]



#Função para aplicar o fillna em um dataframe, buscando os nomes únicos de cada estatística
def custom_fillna(df):
    stat_names = list(set(col.rsplit(' ', 1)[0] for col in df.columns if ' ' in col))
    for stat_name in stat_names[:-2]:
        print(stat_names)
        fillna_logic(df, stat_name)

    return df
