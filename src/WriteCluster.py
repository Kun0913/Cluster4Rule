import os
from tqdm import tqdm
import numpy as np

def write_cluster_data(data, kmeans_labels, cluster_type):
    # 将数据按簇分组
    clusters_data = {}
    for i, label in enumerate(kmeans_labels):
        if label not in clusters_data:
            clusters_data[label] = []
        clusters_data[label].append(data[i])

    # 将每个簇的数据写入到txt文件中
    for cluster_label, cluster_data in tqdm(clusters_data.items(), desc="writing"):
        with open(f'cluster_{cluster_label}_{cluster_type}.txt', 'w') as f:
            for sample in cluster_data:
                f.write(' '.join(map(str, sample)) + '\n')


def write_dbscan_cluster_data(data, dbscan_labels, cluster_type, result_dir):
    # result_dir = 'D:/A_code/Cluster4Rule/result'
    for cluster_id in np.unique(dbscan_labels):
        cluster_data = data[dbscan_labels == cluster_id]
        output_file_path = os.path.join(result_dir, f'cluster_{cluster_id}_{cluster_type}.txt')
        with open(output_file_path, 'w') as f:
            for point in cluster_data:
                f.write(' '.join(map(str, point)) + '\n')

