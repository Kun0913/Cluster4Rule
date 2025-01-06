import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

# 使用绝对路径读取数据集
file_path = 'D:\A_code\Cluster4Rule\data\output_state_1vs1.txt'  # 替换为你的数据集文件的绝对路径
data = pd.read_csv(file_path, header=None, delim_whitespace=True)

# ## 2v2
# # 假设最后两列共同组成标签，其它列是状态量
# X = data.iloc[:, :-2].values
# y = data.iloc[:, -2:].apply(lambda row: '_'.join(row.values.astype(str)), axis=1).values

# # 对标签进行编码
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

## 1v1
# 假设最后一列是动作标签，其它列是状态量
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 对标签进行编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 定义模型
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    # 'SVM': SVC(),
    'Linear SVM': LinearSVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(verbose=0),
    'AdaBoost': AdaBoostClassifier(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis()
}

# 训练和评估每个模型
for model_name, model in tqdm(models.items(), desc="Training Models"):
    # 使用10折交叉验证进行预测
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    # 将预测结果解码回原始标签
    y_decoded = label_encoder.inverse_transform(y)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # 确保标签和预测结果为整数类型
    y_decoded = y_decoded.astype(int)
    y_pred_decoded = y_pred_decoded.astype(int)
    
    # 输出分类报告
    print(f'Classification Report for {model_name}:')
    report = classification_report(y_decoded, y_pred_decoded, target_names=label_encoder.classes_.astype(str))
    print(report)
    print('-' * 80)

# ## 2v2
# for model_name, model in tqdm(models.items(), desc="Training Models"):
#     # 使用10折交叉验证进行预测
#     y_pred = cross_val_predict(model, X, y, cv=10)
    
#     # 将预测结果解码回原始标签
#     y_decoded = label_encoder.inverse_transform(y)
#     y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
#     # 确保标签和预测结果为整数类型
#     y_decoded = y_decoded.astype(int)
#     y_pred_decoded = y_pred_decoded.astype(int)
    
#     # 输出分类报告
#     print(f'Classification Report for {model_name}:')
#     report = classification_report(y_decoded, y_pred_decoded, target_names=label_encoder.classes_.astype(str))
#     print(report)
#     print('-' * 80)