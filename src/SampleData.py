import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import random
import datetime
from SampleDatasetGeneration import generate_sample_data as Sample

feature_ranges = [[0, 180], [0, 180], [-5000, 5000], [0, 350],
                  [0, 5000],[0, 6], [0, 350], [2000, 1500],
                  [0, 1], [0, 1], [0, 1],[0, 1]]
X, y = Sample(n_samples=3000, n_features=12, centers=15, random_state=42,
              feature_ranges=feature_ranges)

# 拼接指定列表到每一行
action_array = [[0, 0], [1, 0], [1, 1], [14,1], [33,0], [30,0]]
action_list = np.array(action_array)
# 将新标签与y的标签进行匹配
new_label_map = {}
for label in set(y):
    new_label_map[label] = random.choice(action_list)
print(new_label_map)

# 找出唯一的类别
unique_classes = np.unique(y)

# 按照y将对应相同的y的X输出
X_with_class = []
for class_label in unique_classes:
    class_indices = np.where(y == class_label)[0]
    class_X = X[class_indices]
    new_label = new_label_map[class_label]
    new_labels = np.full((len(class_X), len(new_label)), new_label)
    class_X_with_new_labels = np.hstack([class_X, new_labels])
    X_with_class.append(class_X_with_new_labels)
X_extended = np.vstack(X_with_class)

# 获取当前系统日期与时间
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# 获取当前脚本文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造 "data" 目录的路径
data_dir = os.path.join(current_dir, "../data")  # 返回上一级目录下的 "data" 目录
# 如果 "data" 目录不存在，则创建它
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# 定义文件名
file_name = os.path.join(data_dir, f"Sample{current_time}.txt")

# 将X按行写入到文件中，数据之间以空格分隔
with open(file_name, 'w') as file:
    for row in X_extended:
        line = ' '.join(map(str, row)) + '\n'
        file.write(line)