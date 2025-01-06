import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

class FuzzyRuleBasedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=2, bandwidth=1.0):
        self.p = p
        self.bandwidth = bandwidth
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in tqdm(self.classes_, desc="Fitting Models"):
            self.models[cls] = X[y == cls]
        # self.models = {cls: X[y == cls] for cls in self.classes_}
        return self

    def _minkowski_kernel(self, distance, bandwidth):
        return np.exp(- (distance / bandwidth) ** self.p)

    def _estimate_membership(self, X, samples):
        memberships = np.zeros(X.shape[0])
        for sample in samples:
            distances = np.linalg.norm(X - sample, ord=self.p, axis=1)
            memberships += self._minkowski_kernel(distances, self.bandwidth)
        return memberships / len(samples)

    def predict(self, X):
        memberships = np.array([self._estimate_membership(X, self.models[cls]) for cls in self.classes_]).T
        return self.classes_[np.argmax(memberships, axis=1)]

    def predict_proba(self, X):
        memberships = np.array([self._estimate_membership(X, self.models[cls]) for cls in self.classes_]).T
        return memberships / memberships.sum(axis=1, keepdims=True)

# 使用绝对路径读取数据集
file_path = 'D:\A_code\Cluster4Rule\data\output_state_1vs1.txt'  # 替换为你的数据集文件的绝对路径
data = pd.read_csv(file_path, header=None, delim_whitespace=True)

# 假设最后两列共同组成标签，其它列是状态量
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# X = data.iloc[:, :-2].values
# y = data.iloc[:, -2:].apply(lambda row: '_'.join(row.values.astype(str)), axis=1).values

# 对标签进行编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 定义模型
models = {
    'Fuzzy Rule Based Classifier': FuzzyRuleBasedClassifier(p=2, bandwidth=1.0),
    # 其他模型...
}

# 训练和评估每个模型
for model_name, model in tqdm(models.items(), desc="Training Models"):
    # 使用10折交叉验证进行预测
    y_pred = cross_val_predict(model, X, y, cv=10)
    
    # 将预测结果解码回原始标签
    y_decoded = label_encoder.inverse_transform(y)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # 输出分类报告
    print(f'Classification Report for {model_name}:')
    report = classification_report(y_decoded, y_pred_decoded, target_names=label_encoder.classes_.astype(str))
    print(report)
    print('-' * 80)