import numpy as np
import pandas as pd
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
