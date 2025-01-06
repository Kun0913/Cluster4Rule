import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 读取数据集
file_path = 'D:\A_code\Cluster4Rule\data\output_state_1vs1.txt'  # 替换为你的数据集文件的绝对路径
data = pd.read_csv(file_path, header=None, delim_whitespace=True)

# 假设最后一列是动作标签，其它列是状态量
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier()

# 训练模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 输出分类报告
report = classification_report(y_test, y_pred, target_names=[f'Label {i}' for i in np.unique(y)])
print(report)