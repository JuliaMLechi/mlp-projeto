import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlp import MLP

# Métrica: acurácia

def acuracia(y_verdadeiro, y_predito):
    """Calcula a acurácia das previsões"""
    predicoes_corretas = np.sum(y_verdadeiro == y_predito)
    return predicoes_corretas / len(y_verdadeiro)

# Métrica: erro quadrático médio (MSE)

def mse(y_verdadeiro, y_predito):
    """Calcula o erro quadrático médio"""
    return np.mean(np.square(y_verdadeiro - y_predito))

# Função de validação cruzada

def validacao_cruzada(x_treino, k_folds, y_treino, model_params):
    """
    Função para Validação Cruzada
    """
    df = pd.DataFrame(x_treino).copy()
    df['target'] = list(y_treino)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    folds = np.array_split(df, k_folds)

    resultados = []
    erros_por_fold = []
    predicoes_por_fold = []

    for i in range(k_folds):
        df_validacao = folds[i]
        df_treino = pd.concat(folds[:i] + folds[i+1:], ignore_index=True)

        x_treino_fold = df_treino.drop(columns='target')
        y_treino_fold = np.vstack(df_treino['target'].values)

        x_validacao_fold = df_validacao.drop(columns='target')
        y_validacao_fold = np.vstack(df_validacao['target'].values)

        modelo = MLP(**model_params)
        modelo.fit(x_treino_fold.values, y_treino_fold)

        y_pred = modelo.predict(x_validacao_fold.values)

        y_validacao_labels = np.argmax(y_validacao_fold, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        acc = acuracia(y_validacao_labels, y_pred_labels)
        erro = mse(y_validacao_fold, y_pred)

        resultados.append(acc)
        erros_por_fold.append(erro)
        predicoes_por_fold.append(list(zip(y_validacao_labels, y_pred_labels)))

        print(f"Fold {i+1}, Acurácia: {acc:.4f}, MSE: {erro:.6f}")

    return resultados, erros_por_fold, predicoes_por_fold


def matriz_confusao(y_true, y_pred, labels=None):
    """
    Parâmetros:
    y_true (list ou array): Rótulos reais.
    y_pred (list ou array): Rótulos previstos.

    Retorna:
    np.ndarray: Matriz de confusão.
    """
    
     #Identificar classes únicas
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(labels)
    
    # Criar um mapeamento de classe para índice
    class_to_index = {label: index for index, label in enumerate(labels)}
    
    # Inicializar a matriz de confusão
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    
    # Preencher a matriz
    for real, pred in zip(y_true, y_pred):
        i = class_to_index[real]
        j = class_to_index[pred]
        matriz[i][j] += 1
    
    # Criar DataFrame para melhor visualização
    df_confusao = pd.DataFrame(matriz, index=labels, columns=labels)
    
    # Calcular métricas
    acuracia = np.trace(matriz) / np.sum(matriz)
    precisao = np.diag(matriz) / np.sum(matriz, axis=0)
    recall = np.diag(matriz) / np.sum(matriz, axis=1)
    f1 = 2 * (precisao * recall) / (precisao + recall)
    
    # Exibir a matriz de confusão
    print("Matriz de Confusão:")
    header = " " * 10 + " ".join(f"{label:^10}" for label in labels)
    print(header)
    for i, row in enumerate(matriz):
        row_str = " ".join(f"{num:^10}" for num in row)
        print(f"{labels[i]:<10}{row_str}")
    
    # Exibir métricas
    print("Métricas por classe:")
    for idx, label in enumerate(labels):
        print(f"Classe {label}:")
        print(f"  Precisão: {precisao[idx]:.2f}")
        print(f"  Recall: {recall[idx]:.2f}")
        print(f"  F1-Score: {f1[idx]:.2f}")
    print(f"\nAcurácia geral: {acuracia:.2f}")
    
    return df_confusao