import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mlp import MLP
from metricas import acuracia, mse, train_test_split_custom, validacao_cruzada, matriz_confusao, gerar_combinacoes_hiperparametros


# Carregar o dataset de caracteres
base_dir = os.path.dirname(__file__)
X = np.load(os.path.join(base_dir, 'X.npy'))
y = np.load(os.path.join(base_dir, 'Y_classe.npy'))

print("Shape original de X:", X.shape)

X = X.reshape(X.shape[0], -1)  # Flatten
print("Shape ajustado de X para a MLP:", X.shape)

# Normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split_custom(X, y, tamanho_teste=0.2)

# Parâmetros da MLP
input_size = X_train.shape[1]
hidden_layers = 5
output_size = 26

grid_hiperparametros = gerar_combinacoes_hiperparametros(input_size, output_size)

print("Iniciando o treinamento na validação")

# Validação cruzada
best_params_por_fold = validacao_cruzada(
    x_treino=X_train,
    k_folds=5,
    y_treino=y_train,
    model_combinacoes_hiper=grid_hiperparametros,
    cross_validation=True
)

# Treino com a melhor combinação

modelo = MLP(**best_params_por_fold)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

y_validacao_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

acc = acuracia(y_validacao_labels, y_pred_labels)
errors = mse(y_validacao_labels, y_pred_labels)

# Gerar plot 
matriz_confusao(y_test, y_pred)
modelo.relatorio_final(errors, X_test, y_test)

print("Treinamento e teste concluídos.")