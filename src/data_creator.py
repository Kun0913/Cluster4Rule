from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def generate_clusters(num_clusters, num_samples_per_cluster, num_features, random_state=None):
    """
    生成具有聚类意义的多个数组
        num_clusters: 聚类的数量
        num_samples_per_cluster: 每个聚类中的样本数量
        num_features: 每个样本的特征数量
        random_state: 随机数生成器的种子值，可选
    返回值：
        X: 生成的数据集
        y: 每个样本所属的聚类标签
    """
    X, y = make_blobs(n_samples=num_samples_per_cluster * num_clusters,
                      n_features=num_features,
                      centers=num_clusters,
                      random_state=random_state)
    return X, y

def plot_clusters(X, y):
    """
    绘制生成的聚类数据

    参数：
        X: 数据集
        y: 每个样本所属的聚类标签
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolors='k')
    plt.title('Generated Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def visualize_tsne(data, kmeans_labels=None, n_components=3):
    """
    使用 t-SNE 进行降维，并可视化数据点在二维或三维空间中的分布。
    参数：
        data: 数据集
        kmeans_labels: 数据点的聚类标签，可选
        n_components: 希望降维的目标维度数，可以是2或3，默认为3
    返回值：
        None
    """
    tsne = TSNE(n_components=n_components, random_state=42)
    X_embedded = tsne.fit_transform(data)

    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for cluster_label in np.unique(kmeans_labels):
            cluster_data = X_embedded[kmeans_labels == cluster_label]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')

        plt.title('t-SNE Visualization of Clustered Data in 2D')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif n_components == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for cluster_label in np.unique(kmeans_labels):
            cluster_data = X_embedded[kmeans_labels == cluster_label]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {cluster_label}')

        ax.set_title('t-SNE Visualization of Clustered Data in 3D')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')
        ax.legend()
        plt.show()
    else:
        raise ValueError("n_components must be 2 or 3")
