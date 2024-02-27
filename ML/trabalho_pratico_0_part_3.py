import numpy as np
import matplotlib.pyplot as plt

# Definindo a semente para reprodutibilidade
np.random.seed(42)

# Gerando dados para a Classe 1
class1 = np.random.randn(20, 2)  # 20 amostras, 2 características

# Gerando dados para a Classe 2 (com alguma diferença nas médias para distinção)
class2 = np.random.randn(20, 2) + np.array([5, 5])  # Deslocamento adicionado

# Visualizando os dados sintéticos
plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Classe 1')
plt.scatter(class2[:, 0], class2[:, 1], color='blue', label='Classe 2')
plt.title('Dados Sintéticos')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.show()
