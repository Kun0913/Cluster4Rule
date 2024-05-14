import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from hopkins import Hopkins
from tqdm import tqdm
from SampleDatasetGeneration import generate_sample_data as Sample

# 获取当前脚本文件所在目录的绝对路径
current_dir = os.getcwd()
# 构造 "data" 目录的路径
data_dir = os.path.join(current_dir, "../data")  # 返回上一级目录下的 "data" 目录
# 定义文件名
file_name = f"Sample20240514133015.txt"
file_path = os.path.join(data_dir, file_name)

# 读取文本文件
with open(file_path, 'r') as file:
    lines = file.readlines()
# 解析每行数据
data = []
for line in lines:
    row = list(map(float, line.strip().split()))
    data.append(row)
# 将数据转换为 NumPy 数组
import numpy as np
data_array = np.array(data)

labels = data_array[:, -2:].tolist()  # 提取最后两列作为标签，并转换为列表
# 创建字典来存储数据分组
grouped_data = {}
for i, label in enumerate(labels):
    label_key = tuple(label)
    if label_key not in grouped_data:
        grouped_data[label_key] = []
    grouped_data[label_key].append(data_array[i, :-2])  # 去除最后两列

for label, group in grouped_data.items():
    grouped_data[label] = np.array(group)

target_label = (0.0, 0.0)
if target_label in grouped_data:
    cluster_data = grouped_data[target_label]
    Hopkins_value = Hopkins(cluster_data)
    print(Hopkins_value)
else:
    Hopkins_value = 0.0

from HybridClustering import hybrid_clustering

if(Hopkins_value > 0.55):

    kmeans_labels = hybrid_clustering(cluster_data,2,6)
    from WriteCluster import write_cluster_data, write_dbscan_cluster_data
    write_cluster_data(cluster_data, kmeans_labels, "kmeans")

    from boundaries import extract_boundaries

    upper_bounds, lower_bounds = extract_boundaries('cluster_0_kmeans.txt')
    print("\n")
    print("kmeans Upper boundaries:", upper_bounds)
    print("kmeans Lower boundaries:", lower_bounds)

    # 使用 t-SNE 进行降维，将数据投影到二维空间
    tsne = TSNE(n_components=2, random_state=42)
    with tqdm(total=100, desc="TSNE") as pbar:
        X_embedded = tsne.fit_transform(cluster_data)
        pbar.update(100)

    # 绘制聚类结果的二维散点图
    plt.figure(figsize=(9, 6))

    # 绘制每个簇的数据点
    for cluster_label in np.unique(kmeans_labels):
        visual_data = X_embedded[kmeans_labels == cluster_label]
        plt.scatter(visual_data[:, 0], visual_data[:, 1], label=f'Cluster {cluster_label}')

    plt.title('t-SNE Visualization of kmeans Clustered Data in 2D')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()



# Hopkins_value = Hopkins(data)
# print(Hopkins_value)
#
# from HybridClustering import hybrid_clustering
# if(Hopkins_value>0.55):
#     kmeans_labels = hybrid_clustering(data,2,6)
#     from WriteCluster import write_cluster_data, write_dbscan_cluster_data
#     write_cluster_data(data, kmeans_labels, "kmeans")
#
# kmeans_labels = hybrid_clustering(data,2,6)
# from WriteCluster import write_cluster_data, write_dbscan_cluster_data
# write_cluster_data(data, kmeans_labels, "kmeans")
#
# from boundaries import extract_boundaries
# upper_bounds, lower_bounds = extract_boundaries('cluster_0_kmeans.txt')
# print("\n")
# print("kmeans Upper boundaries:", upper_bounds)
# print("kmeans Lower boundaries:", lower_bounds)
#
# # 使用 t-SNE 进行降维，将数据投影到二维空间
# tsne = TSNE(n_components=2, random_state=42)
# with tqdm(total=100, desc="TSNE") as pbar:
#     X_embedded = tsne.fit_transform(data)
#     pbar.update(100)
#
# # 绘制聚类结果的二维散点图
# plt.figure(figsize=(9, 6))
#
# # 绘制每个簇的数据点
# for cluster_label in np.unique(kmeans_labels):
#     cluster_data = X_embedded[kmeans_labels == cluster_label]
#     plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')
#
# plt.title('t-SNE Visualization of kmeans Clustered Data in 2D')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.legend()
# plt.show()