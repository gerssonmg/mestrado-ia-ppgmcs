import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_data():
    # Definindo parâmetros para o dataset
    num_amostras = 1000  # Número total de amostras
    dimensoes = 2  # Número de características (features) por amostra

    # Gerando dados aleatórios
    np.random.seed(42)  # Semente para reprodutibilidade
    X = np.random.randn(num_amostras, dimensoes)
    X = np.round(X, 2)  # Arredondando para duas casas decimais

    # Definindo uma função linear simples como limite de decisão
    # y = ax + b, onde a e b são escolhidos arbitrariamente
    a, b = 1.5, -0.1

    # Calculando os rótulos com base na função linear
    # Se y > ax + b, então rótulo 1, caso contrário rótulo 0
    Y = (X[:, 1] > (a * X[:, 0] + b)).astype(int)

    # Convertendo para DataFrame para facilitar manipulação e visualização
    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
    df['Label'] = Y

    return  df
