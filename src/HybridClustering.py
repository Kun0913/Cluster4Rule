import os
os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def hybrid_clustering(data):
    # 使用层次聚类确定簇的数量
    best_score = -1
    best_n_clusters = 0
    for n_clusters in tqdm(range(2, 6), desc="best_n_clusters",leave=False):  # 假设簇的数量在2到5之间
        # 层次聚类
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(data)

        # 使用轮廓系数评估簇的质量
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    print("\n","Best number of clusters:", best_n_clusters)

    # 使用 K-means 进行聚类划分
    kmeans = KMeans(n_clusters=best_n_clusters, n_init='auto')
    with tqdm(total=100, desc="Kmeans") as pbar:
        kmeans_labels = kmeans.fit_predict(data)
        pbar.update(100)


    return kmeans_labels
