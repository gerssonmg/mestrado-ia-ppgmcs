import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Definindo parâmetros para o dataset
num_amostras = 1000  # Número total de amostras
dimensoes = 2  # Número de características (features) por amostra

# Gerando dados aleatórios
np.random.seed(42)  # Semente para reprodutibilidade
X = np.random.randn(num_amostras, dimensoes)
print("X", X)
import pdb

pdb.set_trace()
# Definindo uma função linear simples como limite de decisão
# y = ax + b, onde a e b são escolhidos arbitrariamente
a, b = 1.5, -0.1

# Calculando os rótulos com base na função linear
# Se y > ax + b, então rótulo 1, caso contrário rótulo 0
Y = (X[:, 1] > (a * X[:, 0] + b)).astype(int)

# Convertendo para DataFrame para facilitar manipulação e visualização
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Label'] = Y

# Exibindo as primeiras linhas do dataset
print(df.head(), df['Label'].value_counts())

##### PLOT
# Plotando os pontos
plt.figure(figsize=(10, 6))
plt.scatter(df['Feature_1'][df['Label'] == 0],
            df['Feature_2'][df['Label'] == 0],
            c='red',
            label='Label 0')
plt.scatter(df['Feature_1'][df['Label'] == 1],
            df['Feature_2'][df['Label'] == 1],
            c='blue',
            label='Label 1')

# Adicionando a linha de decisão
x_values = np.linspace(df['Feature_1'].min(), df['Feature_1'].max(),
                       num_amostras)
y_values = a * x_values + b
plt.plot(x_values, y_values, label='Limite de Decisão', color='green')

# Adicionando legendas e rótulos
plt.title('Visualização do Dataset com Limite de Decisão Linear')
plt.xlabel('Feature_1')
plt.ylabel('Feature_2')
plt.legend()
plt.show()
