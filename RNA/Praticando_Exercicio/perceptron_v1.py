from dataset import generate_data
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001

class Perceptron:
    def __init__(self, n_inputs, learning_rate=LEARNING_RATE, n_epochs=100):
        self.weights = np.zeros(n_inputs + 1) # +1 para o bias
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.y_test = 0
        self.X_test = 0
        self.accuracies = []

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # inputs * weights + bias
        import pdb
        # pdb.set_trace()
        return self.activation_function(summation)
    
    def train(self, train_inputs, labels):
        for _ in range(self.n_epochs):
            for inputs, label in zip(train_inputs, labels):
                # print("inputs", inputs)
                # print("label", label)
                # print("weights", self.weights)

                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

            predictions = np.array([self.predict(xi) for xi in self.X_test])
            accuracy = np.mean(predictions == self.y_test)
            self.accuracies.append(accuracy)

            if accuracy >= 0.98:
                print(f"Accuracy: {accuracy:.2f}")
                break
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

def split_data(X, Y, train_size=0.7):
    # Embaralhando os índices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Calculando o número de amostras de treino
    train_samples = int(X.shape[0] * train_size)

    # Dividindo os dados em treino e teste
    X_train = X[indices[:train_samples]]
    Y_train = Y[indices[:train_samples]]
    X_test = X[indices[train_samples:]]
    Y_test = Y[indices[train_samples:]]

    return X_train, X_test, Y_train, Y_test

data = generate_data()

X = data[['Feature_1', 'Feature_2']].values
Y = data['Label'].values

perceptron = Perceptron(n_inputs=2, learning_rate=LEARNING_RATE, n_epochs=100)

# Dividindo os dados em conjuntos de treino e teste manualmente
X_train, X_test, Y_train, Y_test = split_data(X, Y, train_size=0.7)

perceptron.y_test = Y_test
perceptron.X_test = X_test

perceptron.train(X_train, Y_train)

print(X_train)
import pdb
# pdb.set_trace()


def calculate_accuracy(predictions, labels):
    correct_predictions = np.sum(predictions == labels)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Usando o código anterior para gerar previsões
# predictions = [perceptron.predict(x) for x in X_test]

# Testando o Perceptron treinado com o conjunto de teste
predictions = [perceptron.predict(x) for x in X_test]

# Calculando a precisão
accuracy = calculate_accuracy(np.array(predictions), Y_test)

print(f'Precisão: {accuracy:.4f}')