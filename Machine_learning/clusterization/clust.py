import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
%matplotlib inline
from google.colab import files

files.upload()
df = pd.read_csv("blobs.csv")

df.head()

X = np.array(df) # Dataframe -> NumPy array
X[:5]

# KMeans

def plot_k_Means(k,init_method,times = 1):
  jobs = min(k,4)
  for i in range(times):
    #создаем кластеризатор
    kmeans = KMeans(n_clusters = k, init = init_method, n_init = 1, n_jobs = jobs, random_state = 10*i)
    #обучаем кластеризатор
    kmeans.fit(X)
    #создаем график
    plt.figure()
    #рисуем точки
    plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
    plt.ylabel("x_2")
    plt.xlabel("x_1")
    #подписываем график
    plt.title("The number of clusters is " + str(k) + "," + init_method + " initialization")
    
# 2 clusters

plot_k_Means(2,"random",3)
plot_k_Means(2,"k-means++",3)


# AgglomerativeClustering

from sklearn.cluster import AgglomerativeClustering


def plot_AC(k,affinity_method, linkage_method, times = 1):
  for i in range(times):
    #создаем кластеризатор
    model = AgglomerativeClustering(n_clusters = k, affinity = affinity_method, linkage = linkage_method)
    #обучаем кластеризатор
    model.fit(X)
    #создаем график
    plt.figure()
    #рисуем точки
    plt.scatter(X[:,0], X[:,1], c=model.labels_)
    #подписываем график
    plt.ylabel("x_2")
    plt.xlabel("x_1")
    plt.title("The number of clusters is " + str(k) +  ", " + affinity_method + " affinity, " + linkage_method + " linkage")
    
    
# 2 clusters

k = 2
list_of_affinity_methods = ["euclidean", "l1", "l2", "manhattan", "cosine"]
list_of_linkage_methods = ["complete", "average", "single"]
plot_AC(k,"euclidean","ward")
for am in list_of_affinity_methods:
  for lm in list_of_linkage_methods:
    plot_AC(k,am,lm)

# DBSCAN

def plot_DBSCAN(E, ms, metric_method):
  #создаем кластеризатор
  model = DBSCAN(eps = E, min_samples = ms, metric = metric_method)
  #обучаем кластеризатор
  model.fit(X)
  #создаем график
  plt.figure()
  #рисуем точки
  plt.scatter(X[:,0], X[:,1], c=model.labels_)
  #подписываем график
  plt.ylabel("x_2")
  plt.xlabel("x_1")
  plt.title(metric_method + " метод, окрестность " + str(E) + ", точек в окрестности " + str(ms))


# DBSCAN для метрики cityblock

plot_DBSCAN(0.33,5, "cityblock")
plot_DBSCAN(0.28,5, "cityblock")
plot_DBSCAN(0.26,5, "cityblock")
plot_DBSCAN(0.26,3, "cityblock")

# DBSCAN для метрики cosine

plot_DBSCAN(0.02,3, "cosine")
plot_DBSCAN(0.01,5, "cosine")
plot_DBSCAN(0.005,3, "cosine")
plot_DBSCAN(0.005,2, "cosine")

# DBSCAN для метрики euclidean

plot_DBSCAN(0.34,3, "euclidean")
plot_DBSCAN(0.32,3, "euclidean")
plot_DBSCAN(0.26,3, "euclidean")
plot_DBSCAN(0.24,3, "euclidean")

# DBSCAN для метрики l1

plot_DBSCAN(0.3,4, "l1")
plot_DBSCAN(0.24,5, "l1")
plot_DBSCAN(0.26,5, "l1")
plot_DBSCAN(0.2475,4, "l1")

# DBSCAN для метрики l2

plot_DBSCAN(0.34,3, "l2")
plot_DBSCAN(0.32,3, "l2")
plot_DBSCAN(0.26,3, "l2")
plot_DBSCAN(0.24,3, "l2")

# DBSCAN для метрики manhattan

plot_DBSCAN(0.33,5, "manhattan")
plot_DBSCAN(0.28,5, "manhattan")
plot_DBSCAN(0.26,5, "manhattan")
plot_DBSCAN(0.26,3, "manhattan")

