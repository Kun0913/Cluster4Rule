# 创建数据集
from sklearn.datasets import make_blobs
# n_features = 2设置X特征数量
X,y = make_blobs(n_samples = 150, n_features = 2, centers = 3, cluster_std = 0.5, shuffle = True, random_state = 0)

import matplotlib.pyplot as plt
# 使文字可以展示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 使负号可以展示
plt.rcParams['axes.unicode_minus'] = False

plt.scatter(X[:,0], X[:,1], c = 'blue', marker = 'o',  s = 50)
plt.grid()
plt.show()

# 使用k-means进行聚类
from sklearn.cluster import KMeans
# tol = 1e-04设置容忍度
# n_clusters = 3设置的簇的数量
km = KMeans(n_clusters = 3, init = 'k-means++',n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
y_km = km.fit_predict(X)

# k-means++进行聚类
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s = 50, c = 'lightgreen', marker = 's', label = '簇 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s = 50, c = 'orange', marker = 'o', label = '簇 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s = 50, c = 'lightblue', marker = 'v', label = '簇 3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s = 250, marker = '*', c = 'red', label = '中心点')
plt.legend()
plt.grid()
plt.show()


## 使用轮廓图定量分析聚类质量
km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X)

import numpy as np
from matplotlib import cm
# 导入轮廓库
from sklearn.metrics import silhouette_samples

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('簇')
plt.xlabel('轮廓系数')
plt.show()


