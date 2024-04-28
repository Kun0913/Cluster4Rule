# 创建样本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4027)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5,3])*10
df = pd.DataFrame(X, columns = variables, index = labels)
print(df)

# 基于距离矩阵进行层次聚类
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric = 'euclidean')), columns = labels, index = labels)
print(row_dist)

# 聚类正确的方式-1
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
             index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])
# 聚类正确的方式-2
row_clusters = linkage(df.values, method='complete', metric='euclidean')
pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
             index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])

# 画树形图
from scipy.cluster.hierarchy import dendrogram

# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])

row_dendr = dendrogram(row_clusters,
                       labels=labels,
                       # make dendrogram black (part 2/2)
                       # color_threshold=np.inf
                       )
plt.ylabel('欧氏距离')
plt.show()

# 树状图与热力图关联
# 画树状图
fig = plt.figure(figsize = (8,8))
# 两个图形之间的距离（宽和高）、图形本身宽和高
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation = 'left')
# 重新排列数据
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
# 画热力图
axm =fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation = 'nearest', cmap = 'hot_r')

# # 移除树状图的轴
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
        i.set_visible(False)
# 加上颜色棒
fig.colorbar(cax)
# 设置热力图坐标轴
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = None, affinity = 'euclidean', linkage = 'complete',distance_threshold=2.0)
labels = ac.fit_predict(X)
print('Cluster labels: % s' % labels)


