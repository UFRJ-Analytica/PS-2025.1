"""
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

def main():
    try:
        # 1. Definir o caminho do arquivo CSV
        caminho_arquivo = os.path.join(os.path.dirname(__file__), 'Data', 'campeonatos_futebol_atualizacao.csv')
        if not os.path.isfile(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

        # 2. Ler os dados do arquivo CSV
        dados = pd.read_csv(caminho_arquivo)

        # 3. Ajustar nomes das colunas de posse de bola
        colunas = dados.columns.tolist()
        if 'Posse 1(%)' in colunas and 'Posse 2(%)' in colunas:
            dados.rename(columns={
                'Posse 1(%)': 'Posse 1',
                'Posse 2(%)': 'Posse 2'
            }, inplace=True)
        else:
            raise KeyError("Colunas 'Posse 1(%)' e 'Posse 2(%)' não encontradas no arquivo.")

        # 4. Limpar dados ausentes nas colunas essenciais
        colunas_essenciais = ['Posse 1', 'Posse 2', 'Gols 1', 'Gols 2']
        dados.dropna(subset=colunas_essenciais, inplace=True)

        print(f"Quantidade de registros após limpeza: {dados.shape[0]}")

        if dados.empty:
            raise ValueError("Nenhum dado disponível após limpeza.")

        # 5. Converter posse de bola de texto para número decimal
        def converter_posse(valor):
            if isinstance(valor, str):
                return float(valor.replace('%', '')) / 100
            elif isinstance(valor, (int, float)):
                return float(valor)
            else:
                raise ValueError(f"Valor inesperado na posse de bola: {valor}")

        dados['Posse 1'] = dados['Posse 1'].apply(converter_posse)
        dados['Posse 2'] = dados['Posse 2'].apply(converter_posse)

        # 6. Criar coluna de resultado da partida (V = vitória, E = empate, D = derrota)
        def classificar_resultado(gols_casa, gols_visitante):
            if gols_casa > gols_visitante:
                return 'V'  # Vitória
            elif gols_casa == gols_visitante:
                return 'E'  # Empate
            else:
                return 'D'  # Derrota

        dados['Resultado'] = dados.apply(lambda x: classificar_resultado(x['Gols 1'], x['Gols 2']), axis=1)

        # 7. Selecionar as colunas numéricas para usar como características (excluindo gols)
        caracteristicas = dados.select_dtypes(include=[np.number]).drop(columns=['Gols 1', 'Gols 2'])
        alvo = dados['Resultado']

        # 8. Preencher valores ausentes nas características com a média da coluna
        imputador = SimpleImputer(strategy='mean')
        caracteristicas_imputadas = imputador.fit_transform(caracteristicas)

        if caracteristicas_imputadas.size == 0:
            raise ValueError("Nenhuma característica disponível após imputação.")

        # 9. Normalizar as características para melhorar o desempenho dos modelos
        normalizador = StandardScaler()
        caracteristicas_normalizadas = normalizador.fit_transform(caracteristicas_imputadas)

        # 10. Dividir os dados em treino e teste preservando a ordem cronológica
        # Assumindo que os dados estão ordenados cronologicamente no CSV
        tamanho_treino = int(len(dados) * 0.7)
        X_treino = caracteristicas_normalizadas[:tamanho_treino]
        X_teste = caracteristicas_normalizadas[tamanho_treino:]
        y_treino = alvo.iloc[:tamanho_treino]
        y_teste = alvo.iloc[tamanho_treino:]

        # 11. Definir os modelos de classificação a serem treinados e seus parâmetros para GridSearchCV
        modelos = {
            'Regressão Logística': {
                'modelo': LogisticRegression(max_iter=1000),
                'parametros': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'Árvore de Decisão': {
                'modelo': DecisionTreeClassifier(),
                'parametros': {
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Naive Bayes': {
                'modelo': GaussianNB(),
                'parametros': {}
            },
            'Máquina de Vetores de Suporte (SVM)': {
                'modelo': SVC(),
                'parametros': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            }
        }

        resultados = {}
        melhor_modelo_nome = None
        melhor_modelo_score = 0
        melhor_modelo = None

        # 12. Treinar e avaliar cada modelo com GridSearchCV
        for nome, info in modelos.items():
            print(f"\nTreinando modelo: {nome}")
            grid = GridSearchCV(info['modelo'], info['parametros'], cv=5, scoring='accuracy')
            grid.fit(X_treino, y_treino)
            melhor_estimador = grid.best_estimator_
            print(f"Melhores parâmetros: {grid.best_params_}")

            previsoes = melhor_estimador.predict(X_teste)

            acuracia = accuracy_score(y_teste, previsoes)
            precisao = precision_score(y_teste, previsoes, average='weighted', zero_division=0)
            recall = recall_score(y_teste, previsoes, average='weighted', zero_division=0)
            f1 = f1_score(y_teste, previsoes, average='weighted', zero_division=0)

            resultados[nome] = {
                'Acurácia': acuracia,
                'Precisão': precisao,
                'Recall': recall,
                'F1 Score': f1
            }

            print(f"\nRelatório do modelo {nome}:\n")
            print(classification_report(y_teste, previsoes, zero_division=0))

            if acuracia > melhor_modelo_score:
                melhor_modelo_score = acuracia
                melhor_modelo_nome = nome
                melhor_modelo = melhor_estimador

        # 13. Mostrar comparação dos resultados dos modelos
        resultados_df = pd.DataFrame(resultados).T
        print("\nComparação das métricas dos modelos:")
        print(resultados_df.round(3))

        # 14. Exibir gráfico comparativo das métricas
        plt.figure(figsize=(12, 6))
        ax = resultados_df.plot(kind='bar', ax=plt.gca())
        plt.title('Comparação de Modelos de Previsão de Resultados de Futebol')
        plt.ylabel('Pontuação')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 15. Exibir matriz de confusão do melhor modelo
        print(f"\nMatriz de confusão do melhor modelo: {melhor_modelo_nome}")
        previsoes_melhor = melhor_modelo.predict(X_teste)
        matriz_confusao = confusion_matrix(y_teste, previsoes_melhor, labels=['V', 'E', 'D'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Vitória', 'Empate', 'Derrota'],
                    yticklabels=['Vitória', 'Empate', 'Derrota'])
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusão - {melhor_modelo_nome}')
        plt.show()

    except Exception as e:
        print(f"Erro durante a execução: {e}")

if __name__ == '__main__':
    main()
