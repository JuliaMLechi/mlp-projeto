import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlp import MLP
import re

# Métrica: acurácia

def acuracia(y_verdadeiro, y_predito):
    """Calcula a acurácia das previsões"""
    # Se y_verdadeiro for one-hot encoded, converta para índices
    if y_verdadeiro.ndim > 1:
        y_verdadeiro_indices = np.argmax(y_verdadeiro, axis=1)
    else:
        y_verdadeiro_indices = y_verdadeiro  # Se já for um vetor de rótulos, use diretamente

    # Comparando as previsões com as classes verdadeiras
    predicoes_corretas = np.sum(y_verdadeiro_indices == y_predito)
    return predicoes_corretas / len(y_verdadeiro)

# Métrica: erro quadrático médio (MSE)

def mse(y_verdadeiro, y_predito):
    """Calcula o erro quadrático médio"""
    return np.mean(np.square(y_verdadeiro - y_predito))

# Função de validação cruzada

def gerar_combinacoes_hiperparametros(tamanho_entrada, tamanho_saida, epocas=20000):
    """
    Gera uma lista de combinações de hiperparâmetros com taxas de aprendizado e camadas escondidas fixadas internamente.

   Parâmetros:
        tamanho_entrada (int): Tamanho da entrada do modelo.
        tamanho_saida (int): Tamanho da saída do modelo.
        epocas (int): Número de épocas para treinamento. Padrão = 20000.

    Retorno:
        List[Dict]: Lista de dicionários com combinações de hiperparâmetros.
    """
    taxas_aprendizado = [0.01]
    opcoes_neuronios_escondidas = [32]

    combinacoes = []

    for taxa in taxas_aprendizado:
        for camadas in opcoes_neuronios_escondidas:
            parametros = {
                "tamanho_entrada": tamanho_entrada,
                "camadas_escondidas": camadas,
                "tamanho_saida": tamanho_saida,
                "taxa_aprendizado": taxa,
                "epocas": epocas,
                "parada_antecipada": False
            }
            combinacoes.append(parametros)

    return combinacoes

def train_test_split_custom(X, y, tamanho_teste=0.2, random_state=42):
    """
    Divide X e y (one-hot) em conjuntos de treino e teste, embaralhando juntos.

    Retorna: x_train, x_test, y_train, y_test
    """
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(y, columns=[f"target_{i}" for i in range(y.shape[1])])

    df = pd.concat([df_x, df_y], axis=1)

    df_embaralhado = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_teste = int(len(df_embaralhado) * tamanho_teste)
    df_teste = df_embaralhado[:n_teste]
    df_treino = df_embaralhado[n_teste:]

    x_train = df_treino.iloc[:, :X.shape[1]].values
    y_train = df_treino.iloc[:, X.shape[1]:].values

    x_test = df_teste.iloc[:, :X.shape[1]].values
    y_test = df_teste.iloc[:, X.shape[1]:].values

    return x_train, x_test, y_train, y_test


def validacao_cruzada(x_treino, k_folds, y_treino, model_combinacoes_hiper, cross_validation=True):
    """
    Função para Validação Cruzada para one-hot encoded y.
    """
    # Cria DataFrame com colunas para cada classe
    df_X = pd.DataFrame(x_treino)
    df_y = pd.DataFrame(y_treino, columns=[f'target_{i}' for i in range(y_treino.shape[1])])

    df = pd.concat([df_X, df_y], axis=1)

    # Embaralha os dados
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    folds = np.array_split(df, k_folds)

    resultados = []
    erros_por_fold = []
    predicoes_por_fold = []
    best_params_por_fold = []

    for i in range(k_folds):
        df_validacao = folds[i]
        df_treino = pd.concat(folds[:i] + folds[i+1:], ignore_index=True)

        x_treino_fold = df_treino.iloc[:, :x_treino.shape[1]].values
        y_treino_fold = df_treino.iloc[:, x_treino.shape[1]:].values

        x_validacao_fold = df_validacao.iloc[:, :x_treino.shape[1]].values
        y_validacao_fold = df_validacao.iloc[:, x_treino.shape[1]:].values
        
        # extrai índices numéricos para o log
        train_idx = df_treino.index.to_numpy()
        val_idx   = df_validacao.index.to_numpy()

        acc_best = 0
        erro_best = 0
        y_validacao_labels_best = None
        y_pred_labels_best = None
        best_params = None

        for model_params in model_combinacoes_hiper:
            modelo = MLP(**model_params)
            modelo.fit(x_treino_fold, y_treino_fold)

            y_pred = modelo.predict(x_validacao_fold)

            y_validacao_labels = np.argmax(y_validacao_fold, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)

            acc = acuracia(y_validacao_labels, y_pred_labels)
            erro = mse(y_validacao_fold, y_pred)

            if acc > acc_best:
                erro_best = erro
                acc_best = acc
                y_validacao_labels_best = y_validacao_labels
                y_pred_labels_best = y_pred_labels
                best_params = model_params

        resultados.append(acc_best)
        erros_por_fold.append(erro_best)
        predicoes_por_fold.append(list(zip(y_validacao_labels_best, y_pred_labels_best)))
        best_params_por_fold.append(best_params)

        if not cross_validation:
            break
        
        n_epocas = best_params.get('epocas', modelo.epocas)

        print(f"Fold {i+1}/{k_folds}")
        print(f"  • Conjuntos: Train[{train_idx.min()}–{train_idx.max()}], "
            f"Val[{val_idx.min()}–{val_idx.max()}]")
        print(f"  • Erro MSE final: {erro_best:.4f}")
        print(f"  • Taxa de classificação errada: {(1-acc_best)*100:.1f}%"
            f"  →  Acurácia: {acc_best*100:.1f}%\n")

    best_acc_fold = max(resultados)
    idx = resultados.index(best_acc_fold)

    return best_params_por_fold[idx]


def matriz_confusao(y_true, y_pred, labels=None, exibir_plot=True):
    """
    Parâmetros:
    y_true (list ou array): Rótulos reais.
    y_pred (list ou array): Rótulos previstos.

    Retorna:
    np.ndarray: Matriz de confusão.
    """
    
    # Se y_true estiver em one-hot encoding, converte para rótulos
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Se y_pred não estiver em formato de rótulo, converte de probabilidades para rótulos
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Identificar classes únicas
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
    precisao = np.where(np.sum(matriz, axis=0) != 0, precisao, 0)
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
    
    if exibir_plot:
        plt.figure(figsize=(6, 5))
        sns.heatmap(df_confusao, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
        plt.tight_layout()
        plt.show()
        plt.savefig("matriz_confusao.png", dpi=300)
    
    return df_confusao

def plotar_convergencia_erro(erros, salvar_em="convergencia_erro.png", exibir=True):
    """
    Gera o gráfico da convergência do erro ao longo das épocas.

    Parâmetros:
    - erros (list): Lista com os valores do erro em cada época.
    - salvar_em (str): Caminho do arquivo para salvar a imagem do gráfico.
    - exibir (bool): Se True, exibe o gráfico na tela.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(erros, label="Erro por época", color='blue')
    plt.xlabel("Épocas")
    plt.ylabel("Erro quadrático médio")
    plt.title("Convergência do Erro durante o Treinamento")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(salvar_em, dpi=300)

    if exibir:
        plt.show()
        
def _carregar_pesos(path):
    """Lê um arquivo .txt e retorna um vetor 1D de floats."""
    with open(path, 'r') as f:
        text = f.read()
    nums = re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', text)
    return np.array(nums, dtype=float)

def plot_convergencia_pesos(iniciais_path: str, finais_path: str,
                            scatter_path: str = 'convergencia_scatter.png',
                            hist_path: str    = 'convergencia_hist.png'):
    """
    Gera e salva dois gráficos em PNG:
      1) Scatter: Peso Inicial × Peso Final
      2) Histogramas comparativos das distribuições
    """
    # Este código deve estar indentado em 4 espaços, não 5 ou 3
    w_init = _carregar_pesos(iniciais_path)
    w_final = _carregar_pesos(finais_path)

    # Ajuste de tamanho
    if w_init.size != w_final.size:
        min_len = min(w_init.size, w_final.size)
        w_init = w_init[:min_len]
        w_final = w_final[:min_len]

   # Histograma comparativo
    plt.figure(figsize=(7,4))
    plt.hist(w_init, bins=50, alpha=0.6, label='Inicial')
    plt.hist(w_final, bins=50, alpha=0.6, label='Final')
    plt.legend()
    plt.xlabel('Valor do Peso')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Pesos: Antes e Depois do Treino')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()