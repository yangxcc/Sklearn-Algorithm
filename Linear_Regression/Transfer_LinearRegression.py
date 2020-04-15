from sklearn import linear_model
from sklearn.preprocessing import StandardScaler    #引入缩放的包
import matplotlib.pyplot as plt
import numpy as np


def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


data = loadtxtAndcsv_data("data.txt", ",", np.float64)  # 读取数据
X = data[:, 0:-1]  # X对应0到倒数第2列
y = data[:, -1]  # y对应最后一列


# 归一化操作
scaler = StandardScaler()
scaler.fit(X)
x_train = scaler.transform(X)
x_test = scaler.transform(np.array(X))

# 线性模型拟合
model = linear_model.LinearRegression()
model.fit(x_train, y)

# 预测结果
result0 = 289221.547371
result = model.predict(x_test)
print('算法运行结果为',result0)
print('直接调用结果为',result)
x = np.arange(0,len(y))
plt.plot(x,y)
plt.plot(x,result)
plt.show()