"""

Projeto de Previsão de Resultados de Partidas de Futebol - Campeonato Brasileiro 
Este script realiza a previsão de resultados de correspondência usando técnicas de aprendizado de máquina.

Ele carrega os dados, pré-processados, treina vários modelos, executa o ajuste do hiperparâmetro avalia os modelos e apresenta os resultados de forma clara e profissional. 

Colunas do conjunto de dados: (Descrição detalhada das colunas conforme especificado anteriormente) 

Autor: [Seu Nome] Data: [Data Atual]

Descrição das colunas do dataset:
Considerando que “1” se refere ao time da casa e “2” ao time visitante:

Chutes a gol 1 / Chutes a gol 2
Número de finalizações que foram enquadradas (ao menos foram na direção do gol) pelo time 1 / time 2.

Impedimentos 1 / Impedimentos 2
Quantas vezes cada time foi pego em posição de impedimento.

Escanteios 1 / Escanteios 2
Total de cobranças de escanteio a favor de cada equipe.

Chutes fora 1 / Chutes fora 2
Finalizações que não foram na direção do gol (para fora) de cada time.

Faltas 1 / Faltas 2
Quantas faltas cada time cometeu durante a partida.

Cartões amarelos 1 / Cartões amarelos 2
Quantos cartões amarelos foram mostrados a jogadores de cada time.

Cartões vermelhos 1 / Cartões vermelhos 2
Quantos cartões vermelhos foram mostrados a jogadores de cada time.

Cruzamentos 1 / Cruzamentos 2
Número de passes laterais elevados (cruzamentos) realizados por cada equipe.

Laterais 1 / Laterais 2
Quantas vezes cada time executou arremessos laterais.

Chutes bloqueados 1 / Chutes bloqueados 2
Finalizações de cada time que foram bloqueadas por defensores adversários.

Contra-ataques 1 / Contra-ataques 2
Quantas ações de contra-ataque (recuperação e transição rápida) cada equipe conduziu.

Gols 1 / Gols 2
Número de gols marcados por cada time.

Tiro de meta 1 / Tiro de meta 2
Quantos arremessos de meta (goal kicks) cada time cobrou.

Tratamentos 1 / Tratamentos 2
Quantas vezes jogadores de cada time receberam atendimento médico em campo.

Substituições 1 / Substituições 2
Número de trocas de jogadores realizadas por cada equipe.

Tiros-livres 1 / Tiros-livres 2
Quantas cobranças de falta (tiros livres) cada time teve.

Defesas difíceis 1 / Defesas difíceis 2
Número de defesas de alta dificuldade feitas pelos goleiros de cada time.

Posse 1 (%) / Posse 2 (%)
Percentual de tempo de posse de bola de cada equipe ao longo da partida.

Time 1 / Time 2
Nome do time da casa (1) e do time visitante (2).

Posição 1 / Posição 2
Posição tática inicial ou formação de cada equipe (por exemplo: 4-4-2, 3-5-2 etc.).
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully. Total records: {len(data)}")
    return data

"""
Pré processamento dos dados:
Renomeia colunas de posse de bola (Posse 1(%) → Posse 1)
Remove linhas com valores ausentes em colunas essenciais (Posse, Gols)
Converte posse percentual para valores entre 0 e 1
Cria coluna Result com classes W, D, L (Win/Draw/Loss)
"""

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    columns = data.columns.tolist()
    if 'Posse 1(%)' in columns and 'Posse 2(%)' in columns:
        data.rename(columns={
            'Posse 1(%)': 'Posse 1',
            'Posse 2(%)': 'Posse 2'
        }, inplace=True)
    else:
        logger.error("Columns 'Posse 1(%)' and 'Posse 2(%)' not found in dataset.")
        raise KeyError("Columns 'Posse 1(%)' and 'Posse 2(%)' not found in dataset.")

    essential_columns = ['Posse 1', 'Posse 2', 'Gols 1', 'Gols 2']
    data.dropna(subset=essential_columns, inplace=True)
    logger.info(f"Records after removing missing values in essential columns: {len(data)}")

    def convert_possession(value):
        if isinstance(value, str):
            return float(value.replace('%', '')) / 100
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            raise ValueError(f"Unexpected possession value: {value}")

    data['Posse 1'] = data['Posse 1'].apply(convert_possession)
    data['Posse 2'] = data['Posse 2'].apply(convert_possession)

    def classify_result(home_goals, away_goals):
        if home_goals > away_goals:
            return 'W'  # Win
        elif home_goals == away_goals:
            return 'D'  # Draw
        else:
            return 'L'  # Loss

    data['Result'] = data.apply(lambda x: classify_result(x['Gols 1'], x['Gols 2']), axis=1)

    return data

"""
Prepara os dados para o modelo:

Features: todas as colunas numéricas exceto os gols.

Target: coluna Result (rótulo da partida)

Imputa valores ausentes (média)

Escala os dados (normalização padrão)
"""

def prepare_features_and_target(data: pd.DataFrame):
    """Prepare features and target for training."""
    features = data.select_dtypes(include=[np.number]).drop(columns=['Gols 1', 'Gols 2'])
    target = data['Result']

    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    return features_scaled, target

"""
Divide os dados em treino e teste respeitando a ordem cronológica 
(não aleatória), o que simula melhor a previsão real de partidas futuras.
"""

def split_data(X, y, train_ratio=0.7):
    """Split data into training and testing sets preserving chronological order."""
    train_size = int(len(y) * train_ratio)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    logger.info(f"Data split: {len(y_train)} training, {len(y_test)} testing")
    return X_train, X_test, y_train, y_test

def define_models():
    """Define models and their hyperparameters for GridSearch."""
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5]
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {}
        },
        'Support Vector Machine (SVM)': {
            'model': SVC(),
            'params': {
                'C': [1],
                'kernel': ['linear']
            }
        }
    }
    return models



"""
Para cada modelo:

Realiza busca em grade para encontrar os melhores hiperparâmetros

Treina o modelo e gera previsões

Avalia com métricas: accuracy, precision, recall, f1

Salva o melhor modelo baseado na accuracy

""" 

def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    """Train and evaluate models, returning results and best model."""
    results = {}
    best_model_name = None
    best_model_score = 0
    best_model = None

    for name, info in models.items():
        try:
            logger.info(f"Training model: {name}")
            grid = GridSearchCV(info['model'], info['params'], cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
            logger.info(f"Best parameters for {name}: {grid.best_params_}")

            predictions = best_estimator.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }

            logger.info(f"Classification report for {name}:\n{classification_report(y_test, predictions, zero_division=0)}")

            if accuracy > best_model_score:
                best_model_score = accuracy
                best_model_name = name
                best_model = best_estimator
        except Exception as e:
            logger.error(f"Error training model {name}: {e}")

    return results, best_model_name, best_model



"""
Mostra os resultados:

Gráfico de barras comparando as métricas dos modelos

Matriz de confusão do melhor modelo, com labels W, D, L

"""

def display_results(results, best_model_name, best_model, X_test, y_test):
    """Display model results and confusion matrix of the best model."""
    results_df = pd.DataFrame(results).T
    logger.info("Comparison of model metrics:")
    logger.info(f"\n{results_df.round(3)}")

    plt.figure(figsize=(12, 6))
    ax = results_df.plot(kind='bar', ax=plt.gca())
    plt.title('Comparison of Football Match Result Prediction Models')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if best_model is not None:
        logger.info(f"Confusion matrix for best model: {best_model_name}")
        best_predictions = best_model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, best_predictions, labels=['W', 'D', 'L'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Win', 'Draw', 'Loss'],
                    yticklabels=['Win', 'Draw', 'Loss'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.show()
    else:
        logger.warning("No model was successfully trained to display confusion matrix.")

def main():
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'Data', 'campeonatos_futebol_atualizacao.csv')
        data = load_data(file_path)
        data = preprocess_data(data)
        X, y = prepare_features_and_target(data)
        X_train, X_test, y_train, y_test = split_data(X, y)
        models = define_models()
        results, best_model_name, best_model = train_and_evaluate_models(models, X_train, y_train, X_test, y_test)
        display_results(results, best_model_name, best_model, X_test, y_test)
    except Exception as e:
        logger.error(f"Execution error: {e}")

if __name__ == '__main__':
    main()
