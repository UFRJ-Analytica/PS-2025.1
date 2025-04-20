import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Caminho relativo para o CSV
caminho_csv = os.path.join(os.path.dirname(__file__), 'campeonatos_futebol_atualizacao.csv')

# 1. Carregamento dos Dados
df = pd.read_csv(caminho_csv)

# 2. Renomear colunas de posse de bola
df.rename(columns={
    'Posse 1(%)': 'Posse 1',
    'Posse 2(%)': 'Posse 2'
}, inplace=True)

# 3. Remoção de dados faltantes
df.dropna(inplace=True)

# 4. Conversão das colunas de posse de bola para float [0, 1]
# Renomear com segurança (antes de qualquer acesso)
df.rename(columns={
    'Posse 1(%)': 'Posse 1',
    'Posse 2(%)': 'Posse 2'
}, inplace=True)

# Agora sim podemos acessar:
df['Posse 1'] = df['Posse 1'].str.replace('%', '').astype(float) / 100
df['Posse 2'] = df['Posse 2'].str.replace('%', '').astype(float) / 100



df['Posse 1'] = df['Posse 1'].str.replace('%', '').astype(float) / 100
df['Posse 2'] = df['Posse 2'].str.replace('%', '').astype(float) / 100

# 5. Criar variável de resultado do Time 1
def resultado(g1, g2):
    if g1 > g2:
        return 'V'  # Vitória
    elif g1 == g2:
        return 'E'  # Empate
    else:
        return 'D'  # Derrota

df['Resultado'] = df.apply(lambda x: resultado(x['Gols 1'], x['Gols 2']), axis=1)

# 6. Selecionar features numéricas (sem usar gols diretamente)
features = df.select_dtypes(include=[np.number]).drop(columns=['Gols 1', 'Gols 2'])
X = features
y = df['Resultado']

# 7. Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 9. Modelos
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC()
}

# 10. Treinamento e avaliação
resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados[nome] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }
    print(f"\nRelatório para {nome}:\n")
    print(classification_report(y_test, y_pred))

# 11. Comparação final
resultados_df = pd.DataFrame(resultados).T
print("\nMétricas Comparativas:")
print(resultados_df)

# 12. Visualização
plt.figure(figsize=(12, 6))
resultados_df.plot(kind='bar')
plt.title('Comparação de Modelos - ForecastFC')
plt.ylabel('Pontuação')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
