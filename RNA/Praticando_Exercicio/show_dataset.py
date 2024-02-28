import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import generate_data

df = generate_data()

num_amostras = 1000
a, b = 1.5, -0.1

df = generate_data()
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
