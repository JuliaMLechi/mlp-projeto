import numpy as np
import pandas as pd
import random

# Vamos usar o random para inicializar os pesos com valores aleatórios, o que garante que a rede não comece sempre do mesmo jeito

class MLP:
    def __init__(self, tamanho_entrada, camadas_escondidas, tamanho_saida, taxa_aprendizado=0.01, epocas=1000, parada_antecipada=True):
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
        self.parada_antecipada = parada_antecipada

        # Inicialização dos pesos e bias com valores aleatórios entre -1 e 1
        self.pesos_entrada = np.random.uniform(-1, 1, size=(self.tamanho_entrada, self.camadas_escondidas))
        self.pesos_saida = np.random.uniform(-1, 1, size=(self.camadas_escondidas, self.tamanho_saida))
        self.bias_entrada = np.random.uniform(-1, 1, size=(1, self.camadas_escondidas))
        self.bias_saida = np.random.uniform(-1, 1, size=(1, self.tamanho_saida))

        with open("pesosiniciais.txt", "w") as f:
            f.write("Pesos Iniciais:\n")
            f.write(f"{self.pesos_entrada}\n{self.pesos_saida}\n")
            f.write("\nBias Iniciais:\n")
            f.write(f"{self.bias_entrada}\n{self.bias_saida}\n")

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

            if self.parada_antecipada:
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

    def relatorio_final(self, erros, X_test=None, y_test=None):
        """Arquivos de saída úteis para o seu trabalho:
            ● Um arquivo contendo os hiperparâmetros finais da arquitetura da rede neural e
            hiperparâmetros de inicialização. VER
            ● Um arquivo contendo os pesos iniciais da rede. 
            ● Um arquivo contendo os pesos finais da rede.
            ● Um arquivo contendo o erro cometido pela rede neural em cada iteração do
            treinamento.
            ● Um arquivo contendo as saídas produzidas pela rede neural para cada um dos
            dados de teste realizados VER
        """
        # Salvando hiperparâmetros finais
        with open("hiperparametros.txt", "w") as f:
            f.write("Hiperparâmetros Finais e de Inicialização\n")
            f.write(f"Tamanho Entrada: {self.tamanho_entrada}\n")
            f.write(f"Camadas Ocultas: {self.camadas_escondidas}\n")
            f.write(f"Tamanho Saída: {self.tamanho_saida}\n\n")
            f.write(f"Tamanho Taxa de aprendizado: {self.taxa_aprendizado}\n\n")

        # Salvando pesos finais
        with open("pesosfinais.txt", "w") as f:
            f.write("\nPesos Finais:\n")
            f.write(f"{self.pesos_entrada}\n{self.pesos_saida}\n")
            
        # Salvando erro por época
        with open("erro.txt", "w") as f:
            f.write(f"Épocas: {self.epocas}\n")
            f.write("\nErro por Época:\n")
            for epoca, erro in enumerate(erros):
                f.write(f"Época {epoca + 1}: Erro = {erro}\n")

        # Verificando se os dados de teste foram passados
        if X_test is not None and y_test is not None:
            # Verificando se y_test é one-hot encoded ou rótulos diretos
            if y_test.ndim == 1:
                y_true = y_test  # Se já for um vetor de rótulos, usamos diretamente
            else:
                y_true = np.argmax(y_test, axis=1)  # Caso seja one-hot encoded

            # Predição da rede
            y_pred_probs = self.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)

            # Salvando saídas de teste
            with open("saidas_teste.txt", "w") as f:
                for i in range(len(X_test)):
                    f.write(f"Instância {i}:\n")
                    f.write(f"Entrada: {X_test[i]}\n")
                    f.write(f"Classe verdadeira: {y_true[i]}\n")
                    f.write(f"Saída da rede: {y_pred_probs[i]}\n")
                    f.write(f"Classe predita: {y_pred[i]}\n\n")
