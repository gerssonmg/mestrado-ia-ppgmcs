from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Carregando a base de dados IRIS
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Adicionando a coluna de espécies ao dataframe
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Gráfico de pares com Seaborn
sns.pairplot(iris_df, hue='species')
plt.show()
