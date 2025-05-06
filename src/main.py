import pandas as pd
import numpy as np
from mlp import MLP
from metricas import cross_validation

# (1) Carrega dados (exemplo fictício)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# (2) Define parâmetros do modelo
parametros = {
    "tamanho_entrada": X.shape[1],
    "camadas_escondidas": 10,
    "tamanho_saida": 1,
    "taxa_aprendizado": 0.01,
    "epocas": 300
}

# (3) Roda validação cruzada
accs, mses, preds = cross_validation(MLP, X, y, k_folds=5, model_params=parametros)

def relatorio_final(self, erros, nome_arquivo="relatorio_final.txt"):
        """Gera um arquivo de relatório com métricas e pesos finais"""
        with open(nome_arquivo, "w") as f:
            f.write("Relatório Final - MLP\n")
            f.write(f"Épocas: {self.epocas}\n")
            f.write(f"Taxa de Aprendizado: {self.taxa_aprendizado}\n")
            f.write(f"Tamanho Entrada: {self.tamanho_entrada}\n")
            f.write(f"Camadas Ocultas: {self.camadas_escondidas}\n")
            f.write(f"Tamanho Saída: {self.tamanho_saida}\n\n")
            f.write("Pesos Iniciais:\n")
            f.write(f"{self.pesos_entrada}\n{self.pesos_saida}\n")
            f.write("\nBias Iniciais:\n")
            f.write(f"{self.bias_entrada}\n{self.bias_saida}\n")
            f.write("\nErro por Época:\n")
            for epoca, erro in enumerate(erros):
                f.write(f"Época {epoca + 1}: Erro = {erro}\n")
            f.write("\nPesos Finais:\n")
            f.write(f"{self.pesos_entrada}\n{self.pesos_saida}\n")