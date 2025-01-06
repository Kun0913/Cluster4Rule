import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from tqdm import tqdm

# 使用绝对路径读取数据集
file_path = 'D:\A_code\Cluster4Rule\data\output_state_1vs1.txt'  # 替换为你的数据集文件的绝对路径
data = pd.read_csv(file_path, header=None, delim_whitespace=True)

# 假设最后一列是动作标签，其它列是状态量
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

# 训练和评估每个模型
for model_name, model in tqdm(models.items(), desc="Training Models"):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    
    # 输出分类报告
    print(f'Classification Report for {model_name}:')
    report = classification_report(y_test, y_pred, target_names=[f'Label {i}' for i in np.unique(y)])
    print(report)
    print('-' * 80)