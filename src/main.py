import numpy as np
import os
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from mlp import MLP
from metricas import acuracia, mse, train_test_split_custom, validacao_cruzada, matriz_confusao, gerar_combinacoes_hiperparametros, plotar_convergencia_erro


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

print(f"\n Tamanho dos conjuntos:")
print(f"- Treinamento: {X_train.shape}")
print(f"- Teste:       {X_test.shape}")

# Parâmetros da MLP
input_size = X_train.shape[1]
hidden_layers = 5
output_size = 26

# Mostrar hiperparâmetros escolhidos
print("\n Iniciando seleção de hiperparâmetros via validação cruzada")
grid_hiperparametros = gerar_combinacoes_hiperparametros(input_size, output_size)


print("Iniciando o treinamento na validação")

# Validação cruzada
best_params_por_fold = validacao_cruzada(
    x_treino=X_train,
    k_folds=5,
    y_treino=y_train,
    model_combinacoes_hiper=grid_hiperparametros,
    cross_validation=False
)

# Treino com a melhor combinação
inicio = time.time()
modelo = MLP(**best_params_por_fold)
errors = modelo.fit(X_train, y_train) 
fim = time.time()

tempo_treino = fim - inicio
print(f"⏱️ Tempo de treinamento: {tempo_treino:.2f} segundos")
print("Iniciando o teste na validação")

#Teste
y_pred = modelo.predict(X_test)

y_validacao_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

acc = acuracia(y_validacao_labels, y_pred_labels)


# Gerar plot 

matriz_confusao(y_test, y_pred)
modelo.relatorio_final(errors, X_test, y_test)
plotar_convergencia_erro(errors)

print("Treinamento e teste concluídos.")

