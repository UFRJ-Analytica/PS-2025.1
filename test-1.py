import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ===============================
# 1. Caminho seguro para o CSV
# ===============================
csv_path = os.path.join(os.path.dirname(__file__), 'data', 'campeonatos_futebol_atualizacao.csv')
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"⚠️ Arquivo não encontrado: {csv_path}")

# ===============================
# 2. Leitura dos dados
# ===============================
df = pd.read_csv(csv_path)

# ===============================
# 3. Renomear colunas de posse
# ===============================
colunas = df.columns.tolist()
if 'Posse 1(%)' in colunas and 'Posse 2(%)' in colunas:
    df.rename(columns={
        'Posse 1(%)': 'Posse 1',
        'Posse 2(%)': 'Posse 2'
    }, inplace=True)
else:
    raise KeyError("⚠️ As colunas 'Posse 1(%)' e 'Posse 2(%)' não foram encontradas no CSV.")

# ===============================
# 4. Pré-processamento
# ===============================
# Remover valores ausentes
df.dropna(inplace=True)

# Converter porcentagem de posse
df['Posse 1'] = df['Posse 1'].str.replace('%', '').astype(float) / 100
df['Posse 2'] = df['Posse 2'].str.replace('%', '').astype(float) / 100

# Criar variável alvo com base nos gols
def classificar_resultado(g1, g2):
    if g1 > g2:
        return 'V'  # Vitória
    elif g1 == g2:
        return 'E'  # Empate
    else:
        return 'D'  # Derrota

df['Resultado'] = df.apply(lambda x: classificar_resultado(x['Gols 1'], x['Gols 2']), axis=1)

# ===============================
# 5. Preparação dos dados
# ===============================
# Features numéricas (excluindo gols)
X = df.select_dtypes(include=[np.number]).drop(columns=['Gols 1', 'Gols 2'])
y = df['Resultado']

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ===============================
# 6. Modelos e treinamento
# ===============================
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC()
}

resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Avaliação
    resultados[nome] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    # Exibir relatório
    print(f"\n📊 Relatório para {nome}:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

# ===============================
# 7. Resultados finais
# ===============================
resultados_df = pd.DataFrame(resultados).T
print("\n📈 Comparação de Métricas entre os Modelos:")
print(resultados_df.round(3))

# Gráfico comparativo
plt.figure(figsize=(12, 6))
resultados_df.plot(kind='bar')
plt.title('ForecastFC - Comparação de Modelos')
plt.ylabel('Pontuação')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
