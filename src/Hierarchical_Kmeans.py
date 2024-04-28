import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from hopkins import Hopkins
from data_creator import generate_clusters
from tqdm import tqdm

## 生成具有聚类意义的数组
# 参数设置
num_clusters = 3  # 聚类数量
num_samples_per_cluster = 100  # 每个聚类中的样本数量
num_features = 10  # 每个样本的特征数量
random_state = 4027  # 随机数生成器的种子值
# 生成聚类数据
X, y = generate_clusters(num_clusters, num_samples_per_cluster,
                         num_features, random_state)
data = X
print(data, y)

# # 创建示例数据
# np.random.seed(4027)
# data = np.random.rand(100, 10)

Hopkins_value = Hopkins(data)
print(Hopkins_value)

from HybridClustering import hybrid_clustering
kmeans_labels = hybrid_clustering(data)

from WriteCluster import write_cluster_data, write_dbscan_cluster_data
write_cluster_data(data, kmeans_labels, "kmeans")

from sklearn.cluster import DBSCAN
# 使用 DBSCAN 进行聚类
dbscan = DBSCAN(eps=5, min_samples=5)
with tqdm(total=100, desc="DBSCAN") as pbar:
    dbscan_labels = dbscan.fit_predict(data)
    pbar.update(100)
# 获取聚类的标签
write_dbscan_cluster_data(data, dbscan_labels, "DBSCAN")

from boundaries import extract_boundaries
upper_bounds, lower_bounds = extract_boundaries('cluster_0_kmeans.txt')
print("\n")
print("kmeans Upper boundaries:", upper_bounds)
print("kmeans Lower boundaries:", lower_bounds)
upper_bounds, lower_bounds = extract_boundaries('cluster_1_DBSCAN.txt')
print("DBSCAN Upper boundaries:", upper_bounds)
print("DBSCAN Lower boundaries:", lower_bounds)

# 使用 t-SNE 进行降维，将数据投影到二维空间
tsne = TSNE(n_components=2, random_state=42)
with tqdm(total=100, desc="TSNE") as pbar:
    X_embedded = tsne.fit_transform(data)
    pbar.update(100)

# 绘制聚类结果的二维散点图
plt.figure(figsize=(9, 6))

# 绘制每个簇的数据点
for cluster_label in np.unique(kmeans_labels):
    cluster_data = X_embedded[kmeans_labels == cluster_label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')

plt.title('t-SNE Visualization of kmeans Clustered Data in 2D')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()

# 使用 t-SNE 将数据降维到二维
tsne = TSNE(n_components=2)
data_2d = tsne.fit_transform(data)

# 绘制可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.5)
plt.title('t-SNE Visualization of DBSCAN Clustering')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster Label')
plt.show()
