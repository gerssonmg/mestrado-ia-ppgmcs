import matplotlib.pyplot as plt
import matplotlib.patches as patches

def desenhar_perceptron(activado=False):
    fig, ax = plt.subplots()

    # Desenhar neurônios
    entrada_y = [1, 2, 3]  # Posições Y para neurônios de entrada
    for y in entrada_y:
        ax.add_patch(patches.Circle((1, y), 0.1, color='blue', label='Neurônio de Entrada'))

    # Neurônio de saída
    cor_saida = 'red' if activado else 'gray'
    ax.add_patch(patches.Circle((2, 2), 0.1, color=cor_saida, label='Neurônio de Saída'))

    # Desenhar conexões
    for y in entrada_y:
        ax.plot([1.1, 1.9], [y, 2], color='black')

    # Adicionar detalhes
    plt.text(2, 2, 'Ativado' if activado else 'Inativo', horizontalalignment='left', verticalalignment='bottom')
    plt.axis('equal')
    plt.axis('off')
    plt.show()

# Exemplo de uso
desenhar_perceptron(activado=False)  # Para mostrar o neurônio inativo
desenhar_perceptron(activado=True)   # Para mostrar o neurônio ativado
