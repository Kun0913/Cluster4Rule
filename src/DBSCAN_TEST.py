import os
os.environ["OMP_NUM_THREADS"] = '1'

# 创建月半形的数据集
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.manifold import TSNE
import numpy as np

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

from hopkins import Hopkins
test_num = Hopkins(X)
print(test_num)


# 可视化
plt.scatter(X[:,0], X[:,1])
plt.show()

# 比较k-means++聚类， 层次聚类 和 DBSCAN 的区别
# 设置图形
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
# k_means++聚类
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
# 可视化
ax1.scatter(X[y_km==0,0], X[y_km==0,1], c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km==1,0], X[y_km==1,1], c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('K-means')

## 层次聚类
# 将 n_clusters 参数设置为 None。
# 这样做将使算法在每次合并最相似的簇时，自动继续合并直到只剩下一个簇
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
y_ac = ac.fit_predict(X)
# 可视化
ax2.scatter(X[y_ac==0,0], X[y_ac==0,1], c='lightblue', marker='o', s=40, label='cluster 1')
ax2.scatter(X[y_ac==1,0], X[y_ac==1,1], c='red', marker='s', s=40, label='cluster 2')
ax2.set_title('Hierarchical')

plt.legend()
plt.show()

# DBSCAN聚类
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X)

# 可视化
plt.scatter(X[y_db==0,0], X[y_db==0,1], c='blue', marker='o', s=40, label='cluster 1')
plt.scatter(X[y_db==1,0], X[y_db==1,1], c='red', marker='s', s=40, label='cluster 2')
plt.title('DBSCAN')
plt.legend()
plt.show()

# 将数据按簇分组
clusters_data = {}
for i, label in enumerate(y_db):
    if label not in clusters_data:
        clusters_data[label] = []
    clusters_data[label].append(X[i])

# 将每个簇的数据写入到txt文件中
for cluster_label, cluster_data in clusters_data.items():
    
    with open(f'cluster_{cluster_label}.txt', 'w') as f:
        for sample in cluster_data:
            f.write(','.join(map(str, sample)) + '\n')

