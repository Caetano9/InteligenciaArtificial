import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from kneed import DataGenerator, KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score   # ← adicionado DB
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

base = pd.read_csv('Iris.csv', sep=',', encoding='cp1252')

for col in base.columns[:4]:
    Q1 = base[col].quantile(0.25)
    Q3 = base[col].quantile(0.75)
    IQR = Q3 - Q1
    low, up = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    base = base[(base[col] >= low) & (base[col] <= up)]

sns.boxplot(data=base.iloc[:, :4])
plt.title('Boxplot')
plt.show()

Entrada = base.iloc[:, 0:4].values
Entrada = MinMaxScaler().fit_transform(Entrada)

limit = int((Entrada.shape[0] // 2) ** 0.5)
for k in range(2, limit + 1):
    pred = KMeans(n_clusters=k, random_state=0).fit_predict(Entrada)
    print(f'Silhouette Score k = {k}: {silhouette_score(Entrada, pred):.3f}')
    print(f'DB Score k = {k}: {davies_bouldin_score(Entrada, pred):.3f}')  # ← novo

wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=10)
    kmeans.fit(Entrada)
    wcss.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), wcss)
plt.xticks(range(2, 11))
plt.title('The elbow method')
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
print('k (elbowl):', kl.elbow)

kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(Entrada)

plt.scatter(Entrada[labels == 0, 0], Entrada[labels == 0, 1], s=100, c='purple')
plt.scatter(Entrada[labels == 1, 0], Entrada[labels == 1, 1], s=100, c='orange')
plt.scatter(Entrada[labels == 2, 0], Entrada[labels == 2, 1], s=100, c='green')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=120, c='red')
plt.legend(['Cluster 0', 'Cluster 1', 'Cluster 2', 'Centroides'])
plt.show()
