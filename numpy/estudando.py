import numpy as np

##### Criando ARRAYS

# Criar um array de uma lista
a = np.array([1, 2, 3])
print("Array 1D:", a)

# Criar um array 2D (como uma matriz)
b = np.array([[1, 2, 3], [4, 5, 6]])
print("Array 2D:\n", b)

c = np.array([[1, 2, 3], ["A", "B", "C"]])

############
print("\nForma de A:", a.shape)  # (2, 3)
print("Dimensão de A:", a.ndim)  # 2
print("Tipo de dados de A:", a.dtype)  # int64 (pode variar)
print("\nForma de b:", b.shape)  # (2, 3)
print("Dimensão de b:", b.ndim)  # 2
print("Tipo de dados de b:", b.dtype)  # int64 (pode variar)
print("\nForma de C:", c.shape)  # (2, 3)
print("Dimensão de C:", c.ndim)  # 2
print("Tipo de dados de C:", c.dtype)  # int64 (pode variar)

import pdb

# pdb.set_trace()

#### OPERAÇÕES MATEMATICAS
# Soma elementar
c = a + np.array([4, 5, 6])
print("Soma elementar:", c)  # [5 7 9]

# Multiplicação de matrizes
d = np.dot(b, np.array([[1, 4], [2, 5], [3, 6]]))
print("Multiplicação de matrizes:\n", d)

#### FUNÇÕES UNIVERSAIS
# Exemplo: função seno
e = np.sin(np.array([np.pi, np.pi / 2, np.pi / 4]))
print("Seno:", e)

##### Indexação e Fatiamento
# Indexação
print("Primeiro elemento de b:", b[0, 0])  # 1

# Fatiamento
print("Primeira linha de b:", b[0, :])  # [1 2 3]

##### REDIMENSIONAR
# Redimensionar b para 3x2
b_reshape = b.reshape(3, 2)
print("b redimensionado para 3x2:\n", b_reshape)

##### AGREGAÇÕES
print("Máximo de b:", np.max(b))  # 6
print("Média de b:", np.mean(b))  # 3.5

#### STACKING E SPLITING

import numpy as np

# Cria um array com 4 colunas - divisão igual é possível
array_com_4_colunas = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
first, second = np.hsplit(array_com_4_colunas, 2)
print("Primeira metade:\n", first)
print("Segunda metade:\n", second)

# Cria um array com 5 colunas - divisão igual não é possível
#array_com_5_colunas = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
#print("Tentativa de divisão igual entre 5 colunas:\n",
#      np.hsplit(array_com_5_colunas, 2))

# Essa linha causaria um ValueError
#first, second = np.hsplit(array_com_5_colunas, 2)

# Stacking
stacked = np.vstack([a, a * 2])
print("Stacked verticalmente:\n", stacked)

# Splitting
first, second = np.hsplit(stacked, 2)
#print("Primeira metade após split:", first)
