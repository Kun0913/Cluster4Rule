from sklearn.datasets import make_blobs
import numpy as np


def generate_sample_data(n_samples=500, n_features=12, centers=5, cluster_std=1.0, random_state=None,
                          feature_ranges=None):
    # 如果未提供 feature_ranges，默认将所有维度的取值范围设置为 [0, 1]
    if feature_ranges is None:
        feature_ranges = [[0, 1] for _ in range(n_features)]

    # 生成前8维数据
    X, y = make_blobs(n_samples=n_samples, n_features=8, centers=centers, cluster_std=cluster_std,
                      random_state=random_state)

    # 生成后4维数据（只包含0和1）
    X_4d = np.random.randint(2, size=(n_samples, 4))

    # 调整前8维数据的取值范围
    for i in range(8):
        low, high = feature_ranges[i]
        X[:, i] = (X[:, i] / 10.0 + 1) / 2 * (high - low) + low

    # 合并前8维和后4维数据
    X = np.hstack((X, X_4d))

    return X, y

