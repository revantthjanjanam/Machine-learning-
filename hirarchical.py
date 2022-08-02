import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import scipy.cluster._hierarchy as sh
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

data = make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.6,random_state=50)
points = data[0]
#creating dendrogram
dendrogram = dendrogram(linkage(points,method= 'ward'))
plt.show()
#algorithm hirarchical_clustering
hc = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage = 'ward')
y_hc = hc.fit_predict(points)
plt.scatter(points[y_hc==0,0],points[y_hc==0,1],s=100,c='cyan')
plt.scatter(points[y_hc==1,0],points[y_hc==1,1],s=100,c='yellow')
plt.scatter(points[y_hc==2,0],points[y_hc==2,1],s=100,c='red')
plt.scatter(points[y_hc==3,0],points[y_hc==3,1],s=100,c='green')
plt.show()