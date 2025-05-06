import numpy as np
import pandas as pd
import random

# Vamos usar o random para inicializar os pesos com valores aleatórios, o que garante que a rede não comece sempre do mesmo jeito

class MLP:
    def __init__(self, tamanho_entrada, camadas_escondidas, tamanho_saida, taxa_aprendizado=0.01, epocas=1000):
        """
        Inicializa a rede MLP com os parâmetros fornecidos.

        Parâmetros:
        - tamanho_entrada (int): Número de neurônios na camada de entrada.
        - camadas_escondidas (int): Número de neurônios na camada oculta.
        - tamanho_saida (int): Número de neurônios na camada de saída.
        - taxa_aprendizado (float): Taxa de aprendizado para ajuste dos pesos.
        - epocas (int): Número máximo de épocas de treinamento.
        """
        self.epocas = epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.tamanho_entrada = tamanho_entrada
        self.camadas_escondidas = camadas_escondidas
        self.tamanho_saida = tamanho_saida

        # Inicialização dos pesos e bias com valores aleatórios entre -1 e 1
        self.pesos_entrada = np.random.uniform(-1, 1, size=(self.tamanho_entrada, self.camadas_escondidas))
        self.pesos_saida = np.random.uniform(-1, 1, size=(self.camadas_escondidas, self.tamanho_saida))
        self.bias_entrada = np.random.uniform(-1, 1, size=(1, self.camadas_escondidas))
        self.bias_saida = np.random.uniform(-1, 1, size=(1, self.tamanho_saida))

    def funcao_ativacao(self, z):
        """Função de ativação sigmoide"""
        return 1 / (1 + np.exp(-z))

    def funcao_ativacao_derivada(self, z):
        """Derivada da função de ativação sigmoide"""
        sig = self.funcao_ativacao(z)
        return sig * (1 - sig)

    def forward(self, X):
        """Passagem para frente (forward pass)"""
        entrada_oculta = X @ self.pesos_entrada + self.bias_entrada
        saida_oculta = self.funcao_ativacao(entrada_oculta)
        entrada_final = saida_oculta @ self.pesos_saida + self.bias_saida
        y_pred = self.funcao_ativacao(entrada_final)
        return y_pred, entrada_final, entrada_oculta, saida_oculta

    def atualizar_pesos(self, delta_oculta, delta_saida, X, saida_oculta):
        """Atualiza pesos e bias das camadas"""
        self.pesos_saida += self.taxa_aprendizado * (saida_oculta.T @ delta_saida)
        self.bias_saida += self.taxa_aprendizado * np.sum(delta_saida, axis=0, keepdims=True)
        self.pesos_entrada += self.taxa_aprendizado * (X.T @ delta_oculta)
        self.bias_entrada += self.taxa_aprendizado * np.sum(delta_oculta, axis=0, keepdims=True)

    def backward(self, X, erro, entrada_final, entrada_oculta, saida_oculta, y_pred):
        """Propagação para trás (backpropagation)"""
        delta_saida = erro * self.funcao_ativacao_derivada(entrada_final)
        erro_oculto = delta_saida @ self.pesos_saida.T
        delta_oculta = erro_oculto * self.funcao_ativacao_derivada(entrada_oculta)
        self.atualizar_pesos(delta_oculta, delta_saida, X, saida_oculta)

    def fit(self, X, y):
        """Treina a rede com base nos dados de entrada"""
        erros = []
        melhor_erro = np.inf
        paciencia = 10
        epocas_sem_melhora = 0

        for epoca in range(self.epocas):
            y_pred, entrada_final, entrada_oculta, saida_oculta = self.forward(X)
            erro = y - y_pred
            self.backward(X, erro, entrada_final, entrada_oculta, saida_oculta, y_pred)

            perda = np.mean(np.square(erro))
            erros.append(perda)

            if epoca % 100 == 0:
                print(f"Época {epoca}/{self.epocas}, Erro: {perda:.6f}")

            if perda < melhor_erro:
                melhor_erro = perda
                epocas_sem_melhora = 0
            else:
                epocas_sem_melhora += 1

            if epocas_sem_melhora >= paciencia:
                print(f"Parada antecipada na época {epoca}")
                break
        return erros

    def predict(self, X):
        """Realiza a previsão para novos dados"""
        y_pred, _, _, _ = self.forward(X)
        return y_pred

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