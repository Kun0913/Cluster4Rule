import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# 定义一个范围
x = np.linspace(0, 10, 100)

# 创建不同的隶属度函数
gauss_mf = fuzz.gaussmf(x, mean=5, sigma=1)
bell_mf = fuzz.gbellmf(x, a=2, b=4, c=5)
sig_mf = fuzz.sigmf(x, b=5, c=1)

# 绘制隶属度函数
plt.plot(x, gauss_mf, label='Gaussian')
plt.plot(x, bell_mf, label='Bell')
plt.plot(x, sig_mf, label='Sigmoid')
plt.title('Membership Functions')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.legend()
plt.show()