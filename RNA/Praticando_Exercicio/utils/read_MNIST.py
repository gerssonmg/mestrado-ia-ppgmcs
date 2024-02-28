import numpy as np
import os
import struct
from matplotlib import pyplot as plt

def read_mnist(images_path, labels_path):
    with open(labels_path, 'rb') as file:
        magic, num_labels = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.uint8)

    with open(images_path, 'rb') as file:
        magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, rows, cols)

    return images, labels


path_to_dataset = "MNIST_ORG"

train_images_path = os.path.join(path_to_dataset, "train-images.idx3-ubyte")
train_labels_path = os.path.join(path_to_dataset, "train-labels.idx1-ubyte")
train_labels_path = os.path.join(path_to_dataset, "train-labels.idx1-ubyte")
train_labels_path = os.path.join(path_to_dataset, "train-labels.idx1-ubyte")
test_labels_path = os.path.join(path_to_dataset, "t10k-labels.idx1-ubyte")
test_labels_path = os.path.join(path_to_dataset, "t10k-labels.idx1-ubyte")
test_labels_path = os.path.join(path_to_dataset, "t10k-labels.idx1-ubyte")
test_labels_path = os.path.join(path_to_dataset, "t10k-labels.idx1-ubyte")
test_images_path = os.path.join(path_to_dataset, "t10k-images.idx3-ubyte")

x_train, y_train = read_mnist(train_images_path, train_labels_path)
x_test, y_test = read_mnist(test_images_path, test_labels_path)

def visualizar_imagem(index):
    plt.imshow(x_train[index], cmap='gray')
    plt.title(f"Etiqueta: {y_train[index]}")
    plt.show()

# Utilize a função visualizar_imagem aqui
visualizar_imagem(10)  # Exemplo para visualizar a primeira imagem
