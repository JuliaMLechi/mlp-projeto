import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlp import MLP
from metricas import acuracia, mse, validacao_cruzada, matriz_confusao
from sklearn.preprocessing import OneHotEncoder


# Carregar o dataset Iris
iris = load_iris()
X = iris.data  # 4 features por flor
y = iris.target  # 0, 1, 2

# Dividir entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode dos rótulos
onehot_encoder = OneHotEncoder(sparse_output=False)
y_train_reshaped = y_train.reshape(-1, 1)
y_train_onehot = onehot_encoder.fit_transform(y_train_reshaped)

# Definindo os parâmetros da rede
input_size = X_train.shape[1]  # 4
hidden_layers = 5  # Pode ser 5 neurônios escondidos
output_size = len(np.unique(y))  # 3 classes (0, 1, 2)

# Instanciar e treinar a MLP
mlp = MLP(input_size,hidden_layers, output_size, taxa_aprendizado=0.01, epocas=20000)

print("Iniciando o treinamento...")
errors = mlp.fit(X_train, y_train_onehot)

# Fazer predições
y_pred_probs = mlp.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Pegar a classe de maior probabilidade

# Avaliar
acc = acuracia(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {acc * 100:.2f}%")

# Validação Cruzada

validacao_cruzada(x_treino=X_train,
                  k_folds=5, 
                  y_treino=y_train_onehot, 
                  model_params={
        "tamanho_entrada": input_size,
        "camadas_escondidas": hidden_layers,
        "tamanho_saida": output_size,
        "taxa_aprendizado": 0.01,
        "epocas": 20000
    })

matriz_confusao(y_test,y_pred)

# Gerar relatório
mlp.relatorio_final(errors, nome_arquivo="relatorio_final.txt")
print("Treinamento e teste concluídos.")