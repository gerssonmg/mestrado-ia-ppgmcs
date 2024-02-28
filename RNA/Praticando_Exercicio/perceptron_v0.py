import numpy as np

# ENTRADA
BACH = [0, 0]  # COMPOSITOR
BEETHOVEN = [0, 1]  # COMPOSITOR
EINSTEN = [1, 0]  # CIENTISTA
KEPLER = [1, 1]  # CIENTISTA

# SAIDA
COMPOSITOR = 0
CIENTISTA = 1

# Definindo os pontos de entrada e as saídas esperadas
entradas = np.array([KEPLER, BEETHOVEN, EINSTEN, BACH])
saidas_esperadas = np.array([CIENTISTA, COMPOSITOR, CIENTISTA,
                             COMPOSITOR])  # Saídas para A, B, C, D

# Inicializando os pesos e o bias
w = np.zeros(2, int)  # pesos para duas entradas
print("TYPE", type(w[0]))
bias = 0
entrada_bias = 1
taxa_aprendizado = 1


# Função de ativação (função degrau)
def ativacao(x):
  return 1 if x > 0 else 0


# Treinamento do Perceptron
def treinar(entradas, saidas, w, bias, taxa_aprendizado, epocas=10):

    condicao_de_parada = epocas # Definindo a condição de parada
# com base no número de épocas. Mas poderia ser outra coisa, como
# erro mínimo, por exemplo.

    for _ in range(epocas):
        erro_total = 0
        print("Epoca:", _)
        for i in range(len(entradas)):
            print(" ")
            print("Entrada:", i)

            entrada_com_bias = np.append(entradas[i], entrada_bias)  # Adicionando a entrada para o bias
            print("entrada_com_bias", entrada_com_bias)

            pesos_com_bias = np.append(w, bias)
            soma = np.dot(entrada_com_bias, pesos_com_bias)
            print("pesos_com_bias", pesos_com_bias)
            print("V:somatorio", soma)

            saida = ativacao(soma)
            print("saida_obtida", saida)

            erro = saidas[i] - saida
            print("erro", erro)
            # Atualizando os pesos e o bias

            w = w + taxa_aprendizado * erro * entradas[i]
            bias = bias + taxa_aprendizado * erro * entrada_bias
            
            # Some o valor absoluto dos erros para verificar a condição de parada
            erro_total += abs(erro)
            print("erro_total", erro_total)
        
        if erro_total == 0:
            print("Erro total:", erro_total)
            print("rede treinada")
            break

    return w, bias


# Treinando o Perceptron
w_final, bias_final = treinar(entradas, saidas_esperadas, w, bias,
                              taxa_aprendizado)

print("")
print("Pesos finalizados:", w_final)
print("Bias finalizado:", bias_final)


# Testando o Perceptron treinado
def testar(entradas, w, bias):
    entradas_com_bias = np.hstack(
        (entradas, np.ones(
                (entradas.shape[0], 1)
            )
        )
    )  # Adicionando o bias

    saidas = np.array([
        ativacao(np.dot(entrada, np.append(w, bias)))
        for entrada in entradas_com_bias
    ])

    return saidas


saidas = testar(entradas, w_final, bias_final)
print("Saídas do Perceptron treinado :", saidas)
print("Saídas do Perceptron esperadas:", saidas_esperadas)
