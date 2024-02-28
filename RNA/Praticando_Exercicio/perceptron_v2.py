import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gerar dados com 10 variáveis
def generate_data(num_samples=1000, num_features=10):
    np.random.seed(42)  # Para resultados reproduzíveis
    X = np.random.randn(num_samples, num_features)  # Gerar características aleatórias
    X = np.round(X, 2)  # Arredondar para duas casas decimais

    # Gerar rótulos binários (0 ou 1) de forma aleatória
    y = np.random.choice([0, 1], size=num_samples)
    return X, y

# Classe Perceptron
class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(num_features + 1)  # +1 para o viés (bias)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.y_test = 0
        self.X_test = 0
        self.accuracies = []

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Adiciona o bias no início do vetor de entrada
        x = np.insert(x, 0, 1)
        weighted_sum = np.dot(x, self.weights)
        return self.activation(weighted_sum)

    def train(self, X, y):
        for _ in range(self.epochs):
            # print("Epoch:", _)
            # print("Weights:", self.weights)

            for xi, target in zip(X, y):
                # Adiciona o bias no início do vetor de entrada
                xi = np.insert(xi, 0, 1)
                prediction = self.predict(xi[1:])  # Ignora o bias na previsão
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi

            predictions = np.array([self.predict(xi) for xi in self.X_test])
            accuracy = np.mean(predictions == self.y_test)
            self.accuracies.append(accuracy)
            # print(f"Accuracy: {accuracy:.2f}")
            # SHOW PLOT HERE. In each epoch
            # Plotar a acurácia em cada época
            plt.scatter(_, accuracy, c='b')
            plt.text(_, accuracy, f"{accuracy:.2f}", fontsize=9)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Evolution')
            plt.plot(self.accuracies, marker='o', linestyle='-', color='b')
            plt.pause(0.05)  # Pausa breve para atualizar o gráfico
        plt.plot(self.accuracies, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Evolution')
        plt.grid(True)
        plt.show()


# Gerar dados
X, y = generate_data()

# Código para separar os dados em treino e teste e calcular a precisão
# (Isso é apenas um esboço, precisa ser integrado e ajustado conforme necessário)

# Embaralhar os dados
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Dividir os dados
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

perceptron = Perceptron(num_features=10)

perceptron.y_test = y_test
perceptron.X_test = X_test

# Treinar o Perceptron
perceptron.train(X_train, y_train)

# Testar o Perceptron
predictions = np.array([perceptron.predict(xi) for xi in X_test])

# Calcular a precisão
accuracy = np.mean(predictions == y_test)
print(f"Acurácia: {accuracy:.2f}")
