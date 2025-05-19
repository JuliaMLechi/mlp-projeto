# Projeto MLP para Reconhecimento de Caracteres

Este projeto implementa uma Rede Neural Perceptron Multicamadas (MLP) para classificação de caracteres a partir de um conjunto de dados de imagens. O sistema realiza o treinamento da rede, avaliação por métricas como acurácia e matriz de confusão, além de gerar relatórios e arquivos auxiliares para análise dos resultados.

---

## Estrutura do Projeto

- `main.py`  
  Script principal que realiza o carregamento dos dados, pré-processamento, treinamento, predição e avaliação do modelo.

- `mlp.py`  
  Implementação da classe MLP com métodos para inicialização, forward pass, backpropagation, atualização de pesos, treinamento e geração de relatórios.

- `metricas.py`  
  Conjunto de funções para cálculo de métricas como acurácia, erro quadrático médio (MSE), validação cruzada e matriz de confusão.

- Arquivos de dados:  
  - `X.npy` — Dados de entrada (imagens dos caracteres) em formato numpy array.  
  - `Y_classe.npy` — Rótulos/classes correspondentes aos dados de entrada.

- Arquivos gerados durante a execução:  
  - `pesosiniciais.txt` — Pesos iniciais da rede.  
  - `pesosfinais.txt` — Pesos finais após treinamento.  
  - `hiperparametros.txt` — Parâmetros da rede e treinamento.  
  - `erro.txt` — Erro por época durante o treinamento.  
  - `saidas_teste.txt` — Resultados das predições no conjunto de teste.

---

## Descrição dos Arquivos

### main.py

- Carrega os dados do dataset `.npy`.
- Realiza flatten e normalização dos dados.
- Divide os dados em conjunto de treino e teste.
- Inicializa a MLP com parâmetros definidos (tamanho da entrada, número de neurônios na camada oculta, tamanho da saída, taxa de aprendizado, número de épocas).
- Executa o treinamento da rede.
- Realiza predições no conjunto de teste.
- Avalia o modelo usando acurácia e matriz de confusão.
- Gera relatório final da MLP.

---

### mlp.py

Classe `MLP` implementa:

- Inicialização dos pesos e bias aleatoriamente.
- Função de ativação sigmoide e sua derivada.
- Forward pass: cálculo da saída da rede.
- Backpropagation: atualização dos pesos usando o erro calculado.
- Método `fit` para realizar o treinamento em múltiplas épocas, com opção de parada antecipada.
- Método `predict` para realizar predições.
- Método `relatorio_final` para salvar em arquivos os pesos, hiperparâmetros, erro por época e resultados dos testes.

---

### metricas.py

Funções para avaliar o desempenho do modelo:

- `acuracia(y_verdadeiro, y_predito)` — Calcula a proporção de predições corretas.
- `mse(y_verdadeiro, y_predito)` — Calcula o erro quadrático médio.
- `validacao_cruzada(...)` — Executa validação cruzada K-fold para o modelo MLP.
- `matriz_confusao(y_true, y_pred, labels=None)` — Gera e exibe a matriz de confusão, com visualização gráfica.

---

## Requisitos

- Python 3.x
- Bibliotecas:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn

Você pode instalar as dependências com:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn