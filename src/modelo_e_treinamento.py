import seaborn as sns
import matplotlib as plt
from typing import Tuple, Union, Dict
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np

# ========================
#   Funções de modelos
# ========================

def get_xgboost_model(X_train: Union[pd.DataFrame, np.ndarray], 
                    y_train: Union[pd.Series, np.ndarray]
                    ) -> XGBClassifier:
    """
    Faz tuning e treina um XGBoost com RandomizedSearchCV.

    Parâmetros:
    - X_train (DataFrame ou ndarray): Dados de treino.
    - y_train (Series ou ndarray): Rótulos.

    Retorna:
    - XGBClassifier treinado com os melhores hiperparâmetros.
    """


    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    rand_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42
    )
    rand_search.fit(X_train, y_train)

    print("\n[LOG - XGBoost] Melhores hiperparâmetros encontrados:")
    for param, value in rand_search.best_params_.items():
        print(f"  {param}: {value}")

    return rand_search.best_estimator_



def get_mlp_model(X_train: Union[pd.DataFrame, np.ndarray], 
                y_train: Union[pd.Series, np.ndarray], 
                hidden_layers: Tuple[int, ...] = (64, 32),
                alpha: float = 1e-4,
                max_iter: int = 100
                ) -> Pipeline:
    """
    Treina uma MLP com StandardScaler dentro de um pipeline.

    Parâmetros:
    - X_train (DataFrame ou ndarray): Dados de treino.
    - y_train (Series ou ndarray): Rótulos.
    - hidden_layers (Tuple[int, ...]): Arquitetura da rede.
    - alpha (float): Termo de regularização L2.
    - max_iter (int): Número máximo de épocas.

    Retorna:
    - Pipeline sklearn com scaler e MLP treinada.
    """


    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate='adaptive',
            max_iter=max_iter,
            random_state=42,
            verbose=True
        ))
    ])
    mlp_pipeline.fit(X_train, y_train)

    print("\n[LOG - MLP] Arquitetura usada:")
    print(f"  hidden_layer_sizes: {hidden_layers}")
    print(f"  max_iter: {max_iter}")

    return mlp_pipeline


def build_ensemble(xgb_model: XGBClassifier, mlp_model: Pipeline) -> VotingClassifier:
    """
    Cria um ensemble soft voting entre XGBoost e MLP.

    Parâmetros:
    - xgb_model (XGBClassifier): Modelo XGBoost treinado.
    - mlp_model (Pipeline): Pipeline com MLP treinada.

    Retorna:
    - VotingClassifier com soft voting.
    """

    ensemble = VotingClassifier(estimators=[
        ('xgb', xgb_model),
        ('mlp', mlp_model)
    ], voting='soft')
    return ensemble



# ===========================
#   Funções de treinamento
# ===========================

def split(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa features e target e realiza o split em treino e teste.

    Parâmetros:
    - df (pd.DataFrame): DataFrame completo com features e coluna alvo.
    - target_column (str): Nome da coluna alvo.
    - test_size (float, opcional): Proporção dos dados para teste (padrão: 0.2).
    - random_state (int, opcional): Semente aleatória para reprodutibilidade (padrão: 42).

    Retorna:
    - Tuple: (X_train, X_test, y_train, y_test)
    """

    if target_column not in df.columns:
        raise ValueError(f"A coluna '{target_column}' não existe no DataFrame.")

    X = df.drop(columns=[target_column])
    y = df['Target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def ensemble_train(X_train: Union[pd.DataFrame, np.ndarray],
                y_train: Union[pd.Series, np.ndarray]
                ) -> VotingClassifier:
    """
    Treina os modelos base e o ensemble.

    Retorna:
    - VotingClassifier treinado.
    """
    print("Treinando XGBoost...")
    xgb_model = get_xgboost_model(X_train, y_train)

    print("Treinando MLP...")
    mlp_model = get_mlp_model(X_train, y_train)

    print("Treinando ensemble...")
    ensemble = build_ensemble(xgb_model, mlp_model)
    ensemble.fit(X_train, y_train)

    return ensemble




def ensemble_test(ensemble_model: VotingClassifier,
                X_test: Union[pd.DataFrame, np.ndarray]
                ) -> np.ndarray:
    """
    Realiza a predição do ensemble.

    Retorna:
    - y_pred (np.ndarray): Predições.
    """
    return ensemble_model.predict(X_test)


def ensemble_evaluate(y_test: Union[pd.Series, np.ndarray],
                    y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    Avalia o desempenho do ensemble e exibe matriz de confusão com seaborn.
    """
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    print("\n==== Avaliação ====")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot da matriz de confusão com seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão Normalizada")
    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}



def train_and_evaluate_xgboost(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    random_state: int = 42
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    Treina um modelo XGBoost com RandomizedSearchCV e avalia nos dados de teste.

    Parâmetros:
    - X_train, y_train: dados de treino
    - X_test, y_test: dados de teste
    - random_state: controle de aleatoriedade

    Retorna:
    - Dicionário com accuracy, classification report e matriz de confusão
    """

    print("[XGBoost] Iniciando tuning...")

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    rand_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=random_state
    )
    rand_search.fit(X_train, y_train)

    print("\n[LOG - XGBoost] Melhores hiperparâmetros:")
    for param, value in rand_search.best_params_.items():
        print(f"  {param}: {value}")

    best_model = rand_search.best_estimator_

    # === Teste ===
    print("\n[LOG - XGBoost] Realizando predição...")
    y_pred = best_model.predict(X_test)

    # === Avaliação ===
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    print("\n==== Avaliação ====")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão com seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=True, yticklabels=True)
    plt.title("Matriz de Confusão (Normalizada)")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}

if __name__ == "__main__":
# Import a ser utilizado nos notebooks:
#    from modelo_e_treinamento import (
#    split,
#   ensemble_train,
#    ensemble_test,
#    ensemble_evaluate
#)



    csv_path = " "

    
    print("Splitando dados...")
    X_train, X_test, y_train, y_test = split(csv_path, target_column="target")

    # === Treinamento do Ensemble ===
    print("\nTreinamento o ensemble...")
    ensemble_model = ensemble_train(X_train, y_train)

    # === Teste ===
    print("\nTestando o ensemble...")
    y_pred = ensemble_test(ensemble_model, X_test)

    # === Avaliação ===
    print("\nAvaliação do modelo:")
    metrics = ensemble_evaluate(y_test, y_pred)

