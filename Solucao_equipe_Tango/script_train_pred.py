# Script de treinamento e predição usando o modelo no jupyter notebook solução nº2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.preprocessing import LabelEncoder,StandardScaler
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
from itertools import combinations
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import time
from imblearn.over_sampling import SMOTE

def remove_outliers(df, filter_cols, lowq, supq):
    # Iterate over each numerical column
    for col in filter_cols:
        # Calculate the IQR
        Q1 = df[col].quantile(lowq)
        Q3 = df[col].quantile(supq)
        IQR = Q3 - Q1
        # Define the upper and lower bounds for outliers
        fac = 1.5
        lower_bound = Q1 - fac * IQR
        upper_bound = Q3 + fac * IQR
        #print(f"{col} - Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
        # Remove outliers
        #print((df[col] <= upper_bound))
        df = df[( (df[col] <= upper_bound) | (df[col].isna()) ) ] # (df[col] >= lower_bound) & 
    #print('\n')
    return df

# Leitura dos dados
train_df_path= "Data/ace_df_treino.csv"
test_df_path = "Data/ace_df_teste.csv"

labeled_test = 0

ace_df = pd.read_csv(train_df_path, encoding='latin-1')
ace_df_teste = pd.read_csv(test_df_path, encoding='latin-1')


ace_df['Resultado Time 1'] = ace_df.apply(lambda row: '0' if row['Gols 1'] > row['Gols 2'] else '1', axis=1)


## Remoção dos outliers usando IQR em função própria.
ace_df = remove_outliers(ace_df , ['Faltas 2', 'Chutes fora 2'], 0.0, 0.75)
ace_df = remove_outliers(ace_df , [ 'Chutes a gol 1', 'Chutes a gol 2'], 0.0, 0.75)



## Preenchimento de valores nulos em certas colunas.

# Caso uma coluna tenha mais do que k % de linhas nulas, ela é desconsiderada. Foram realizadas tentativas de utilizar algumas dessas colunas, mas chegamos à conclusão que a introdução de uma quantidade muito alta de dados artificiais traz viés ao modelo e não agrega.
# Dessa forma, colunas com menos do que k % de linhas nulas tem seus valores nulos preenchidos com a mediana da coluna.
null_map = ace_df.isna().sum()
numeric_columns = ace_df.select_dtypes(include='number').columns

threshold = 0.5 * len(ace_df)
for col in numeric_columns:
    if (null_map[col] < threshold):
        ace_df[col] = ace_df[col].fillna(ace_df[col].median(numeric_only= True))


## Codificação de colunas não numéricas
label_encoder_dict = {}

# Itere sobre as colunas do dataframe
for column in ace_df.columns:
    # Verifique se a coluna é do tipo 'object'
    if (ace_df[column].dtype == 'object') and (column != 'Position 1')and (column != 'Position 2'):
        label_encoder_dict[column] = LabelEncoder()

        # Aplique a codificação usando o LabelEncoder
        ace_df[column] = label_encoder_dict[column].fit_transform(ace_df[column])


## Remoção efetiva de colunas com nulos e seleção de variáveis preditoras.
null_columns = ace_df.columns[ace_df.isnull().any()].tolist()


ace_df = ace_df.drop(null_columns,axis = 1) #[selected_variables]
print(ace_df.columns)

selected_var = ['Chutes a gol 1', 'Chutes a gol 2', 'Escanteios 1', 'Escanteios 2',
       'Chutes fora 1', 'CartÃµes amarelos 1', 'CartÃµes vermelhos 1',
       'CartÃµes vermelhos 2', 'Laterais 1', 'Time 1']
       
ace_df = ace_df[ selected_var + ['Gols 1', 'Gols 2', 'Resultado Time 1']]


## Undersampling para rebalanceamento de classes no dataSet. Isso é feito para o modelo não ficar inviesado para das classes.

count_per_class = ace_df.groupby('Resultado Time 1').size()

samples_threshold = min(ace_df.groupby('Resultado Time 1').size())

for mat, count in count_per_class.items():
    while count > samples_threshold:
        # Pega índices das linhas no grupo 'mat' até atingir o limite
        exceding_samples = ace_df[ace_df['Resultado Time 1'] == mat].sample(n=count - samples_threshold).index

        # Faça o drop das linhas excedentes
        ace_df = ace_df.drop(exceding_samples)

        # Atualize a contagem para o grupo 'genre'
        count = len(ace_df[ace_df['Resultado Time 1'] == mat])


## Preparação dos dados

target_var1 = ['Gols 1']
target_var2 = ['Gols 2']
target_var3 = ['Resultado Time 1']

X = ace_df.drop(target_var1+target_var2+target_var3, axis=1)
y = ace_df[target_var3]

xgb_classifier_model = XGBClassifier(subsample=1.0, reg_lambda=0.1, reg_alpha=0, n_estimators=50, max_depth=3, learning_rate=0.45, gamma=1, colsample_bytree=0.8)


## Cálculo de métricas do modelo usando Validação Cruzada com 10 folds.
scores = cross_validate(xgb_classifier_model, X, y, cv=10, scoring=['accuracy', 'precision', 'recall', 'roc_auc'])
mean_accuracy = scores['test_accuracy'].mean()
mean_precision =scores['test_precision'].mean()
mean_recall = scores['test_recall'].mean()
mean_roc_auc = scores['test_roc_auc'].mean()

print("Mean Accuracy:", mean_accuracy)
print("Mean Precision:", mean_precision)
print("Mean Recall: ", mean_recall)
print("Mean Roc_auc: ", mean_roc_auc)


### Predição dos dados de teste importados


## Tratamento de nulos, substituindo pela mediana para as variáveis preditoras
numeric_columns = ace_df_teste.select_dtypes(include='number').columns

for col in numeric_columns:
    if col in selected_var:
        ace_df_teste[col] = ace_df_teste[col].fillna(ace_df_teste[col].median(numeric_only= True))


## Codificação de variáveis categóricas, dentre as preditoras
for col in selected_var:
    if (ace_df_teste[col].dtype == 'object') and (col != 'Position 1') and (col != 'Position 2'):
        #print(col)
        le = label_encoder_dict[col]
        unseen = set(ace_df_teste[col].unique()) - set(le.classes_)
        if unseen:
            le.classes_ = np.append(le.classes_, list(unseen))
        ace_df_teste[col] = le.transform(ace_df_teste[col])


## Preparação de dados de teste para predição
X_test = ace_df_teste.drop(target_var1+target_var2+target_var3, axis=1)[selected_var]

if(labeled_test):
    y_test = ace_df_teste[target_var3[0]].astype(int)


## Criação e treinamento de um modelo XGB Classifier
xgb_classifier_model = XGBClassifier(subsample=1.0, reg_lambda=0.1, reg_alpha=0, n_estimators=50, max_depth=3, learning_rate=0.45, gamma=1, colsample_bytree=0.8)

xgb_classifier_model.fit(X, y)

## Predição de valores de teste.
y_teste_pred = xgb_classifier_model.predict(X_test)


## Caso seja um teste de dados já classificados corretamente, plotar matriz confusão. Senão, escrever resultados em um arquivo.

if (labeled_test):
    
    # Garante que ambos os arrays sejam 1D
    labels = np.unique(np.concatenate([np.ravel(y_test), np.ravel(y_teste_pred)]))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_teste_pred, labels=labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # DataFrames para visualização
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_percent_df = pd.DataFrame(cm_percent, index=labels, columns=labels)

    # Anotações combinadas
    annot = cm_df.astype(str) + "\n" + cm_percent_df.round(1).astype(str) + "%"

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão dos dados de Teste')
    plt.tight_layout()
    plt.savefig('Data/confusion_matrix_test.png')  # salva o gráfico
    print('Matrix confusão salva!')

    score = accuracy_score(y_test, y_teste_pred)
    score
    print(f"Accuracy: {score}")
else:
    result_df = pd.DataFrame({f'predicted - {target_var3[0]}': y_teste_pred}, index = ace_df_teste.index)
    result_df.to_csv('Data/resultado_predicao.csv', header=True)




