from __future__ import print_function
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing  import LabelEncoder
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix, classification_report

le = LabelEncoder()
df = pandas.read_csv('Iris2.csv')


#transformando os valores da coluna 'class' em números
le.fit(df['class'])
df['class'] = le.transform(df['class'])



##Aplicando o k-means em um gráfico 3d 

iris_matrix = pandas.DataFrame.to_numpy(df[['sepallength','sepalwidth','petallength','petalwidth']])

km = KMeans(n_clusters=3, init='random') ##aplicando sem nenhum hyperparameter 
clusters = km.fit_predict(df) 

fig = plt.figure() 

ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(km.cluster_centers_[:, 3],
          km.cluster_centers_[:, 0],
          km.cluster_centers_[:, 2],
          s = 250,
          marker='o',
          c='red',
          label='centroids')

#definindo labels do gráfico, número de clusters, , colorscheme
scatter = ax.scatter(df['petalwidth'], df['sepallength'], df['petallength'], c=clusters, s=20, cmap='winter')

ax.set_title('K-Means Clustering')
ax.set_xlabel('Petal Width')
ax.set_ylabel('Sepal Length')
ax.set_zlabel('Petal Length')
ax.legend()
plt.show()

"""
#transformando os valores da coluna 'class' em números
le.fit(df['class'])
df['class'] = le.transform(df['class'])
                                                
##Aplicando kmeans e mostrando um grafico do resultado
#transformando os dados em uma matrix                                      
iris_matrix = pandas.DataFrame.to_numpy(df[['sepallength','sepalwidth','petallength','petalwidth']])

#aplicando o kmeans                                  
cluster_model = KMeans(n_clusters=3, init='k-means++', random_state=10)
cluster_model.fit(iris_matrix)
cluster_lables = cluster_model.fit_predict(iris_matrix)
kmeans = df
kmeans["especies"] = cluster_lables

#Plotando o grafico do kmeans               coordenadas         x              y
sns.FacetGrid(df, hue="class", height=7).map(plt.scatter, "sepallength", "sepalwidth", edgecolor ="w").add_legend()
plt.show(block=True)           
""" 
                                       
                        
                           
                           
##calculando a precisão do score                  
#print(sm.accuracy_score(df["class"], cluster_model.labels_))

#calculando a matriz de confusão
#print(pandas.crosstab(df["class"], cluster_model.labels_))


"""
#sns.scatterplot(data=df, x="class", y="sepallength", hue=kmeans.labels_)
#plt.show()
"""
"""
#convertendo para matrix
iris_matrix = pandas.DataFram

#transformando os valores da coluna 'class' em números
le.fit(df['class'])
df['class'] = le.transform(df['class'])


#aplicando o kmeans 
kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
sns.scatterplot(data=df, x="class", y="sepallength", hue=kmeans.labels_)
plt.show()

"""

"""
#tive que apagar a coluna class do arquivo original
df = pandas.read_csv('Iris.csv')

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(df)
sns.scatterplot(data=df, x="sepallength", y="sepalwidth", hue=kmeans.labels_)
plt.show()

"""

"""
x = numero de instancia
y = sepallength
"""

"""
#print(df.describe())
#sns.boxplot(data=df, x='sepallength')
#plt.show()
sns.boxplot(data=df, x='sepalwidth')
plt.show()
sns.boxplot(data=df, x='petallength')
plt.show()
sns.boxplot(data=df, x='petalwidth')
plt.show()
"""

"""
x = data.values 

#data.head(3)
visualizer = KElbowVisualizer(KMeans(3), k=(1,3)).fit(data)
visualizer.show()
##print(data)


sns.scatterplot(data, x="var1", y="var2", hue=kmeans.labels_)
plt.show()
"""

