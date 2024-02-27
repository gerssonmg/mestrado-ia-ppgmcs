from sklearn.datasets import load_iris
import pandas as pd

# Carregando a base de dados IRIS
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Visualizando as primeiras linhas
print(iris_df.head())

# Obtendo estatísticas descritivas
print(iris_df.describe())
