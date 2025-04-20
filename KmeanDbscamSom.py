import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from kneed import KneeLocator
from sklearn.cluster import KMeans, DBSCAN                             # ← DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from minisom import MiniSom                                             

base = pd.read_csv('Iris.csv', sep=',', encoding='cp1252')

for col in base.columns[:4]:
    Q1 = base[col].quantile(0.25)
    Q3 = base[col].quantile(0.75)
    IQR = Q3 - Q1
    low, up = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    base = base[(base[col] >= low) & (base[col] <= up)]

Entrada = MinMaxScaler().fit_transform(base.iloc[:, 0:4].values)

limit = int((Entrada.shape[0] // 2) ** 0.5)
for k in range(2, limit + 1):
    pred = KMeans(n_clusters=k, random_state=0).fit_predict(Entrada)
    print(f'Silhouette k={k}: {silhouette_score(Entrada, pred):.3f}  '
          f'DB k={k}: {davies_bouldin_score(Entrada, pred):.3f}')

wcss = []
for i in range(2, 11):
    wcss.append(KMeans(n_clusters=i, random_state=10).fit(Entrada).inertia_)
kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
print('k (elbow):', kl.elbow)

kmeans = KMeans(n_clusters=3, random_state=0)
labels_km = kmeans.fit_predict(Entrada)
print('KMeans grupos:', len(np.unique(labels_km)))

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_db = dbscan.fit_predict(Entrada)
clusters_db = set(labels_db) - {-1}
print('DBSCAN grupos (sem ruído):', len(clusters_db))

som = MiniSom(x=1, y=3, input_len=4, sigma=0.5, learning_rate=0.5,
              random_seed=0)
som.random_weights_init(Entrada)
som.train_random(Entrada, 100)
labels_som = np.array([som.winner(x)[1] for x in Entrada])   
print('SOM grupos:', len(np.unique(labels_som)))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].scatter(Entrada[:, 0], Entrada[:, 1], c=labels_km, cmap='viridis', s=20)
ax[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              c='red', s=60, marker='X')
ax[0].set_title('KMeans (k=3)')

mask = labels_db != -1
ax[1].scatter(Entrada[mask, 0], Entrada[mask, 1], c=labels_db[mask],
              cmap='viridis', s=20)
ax[1].set_title('DBSCAN (eps=0.3)')

ax[2].scatter(Entrada[:, 0], Entrada[:, 1], c=labels_som, cmap='viridis', s=20)
ax[2].set_title('SOM (1×3)')

plt.tight_layout()
plt.show()
