import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 最大最小值归一化
def MMS(data):
    return MinMaxScaler().fit_transform(data)


# 标准化
def SS(data):
    return StandardScaler().fit_transform(data)


# 均值中心化
def CT(data):
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data


# 标准正态变换
def SNV(data):
    m = data.shape[0]
    n = data.shape[1]
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = np.array([[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)])
    return data_snv


# 移动平均平滑
def MA(a, WSZ=21):
    for i in range(a.shape[0]):
        out0 = np.convolve(a[i], np.ones(WSZ, dtype=int), 'valid') / WSZ  # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(a[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(a[i, :-WSZ:-1])[::2] / r)[::-1]
        a[i] = np.concatenate((start, out0, stop))
    return a


# Savitzky-Golay平滑滤波
def SG(data, w=21, p=3):
    return signal.savgol_filter(data, w, p)


# 一阶导数
def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di


# 二阶导数
def D2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di


# 趋势校正(DT)
def DT(data):
    x = np.asarray(range(350, 2501), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)
    return out


# 多元散射校正
# MSC(数据)
def MSC(Data):
    # 计算平均光谱
    n, p = Data.shape
    msc = np.ones((n, p))

    for j in range(n):
        mean = np.mean(Data, axis=0)

    # 线性拟合
    for i in range(n):
        y = Data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return msc

# #载入数据
# data_path = './/data//corn//m5.csv' #数据
# xcol_path = './/data//corn//label.csv' #波长
# data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
# xcol = np.loadtxt(open(xcol_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
#
# # 绘制MSC预处理后图片
# plt.figure(500)
# x_col = xcol  #数组逆序
# y_col = np.transpose(data)
# plt.plot(x_col, y_col)
# plt.xlabel("Wavenumber(nm)")
# plt.ylabel("Absorbance")
# plt.title("The spectrum of the raw for dataset",fontweight= "semibold",fontsize='large') #记得改名字MSC
# plt.show()
#
# #数据预处理、可视化和保存
# datareprocessing_path = './/data//dataMSC.csv' #波长
# Data_Msc = MSC(data) #改这里的函数名就可以得到不同的预处理
#
# # 绘制MSC预处理后图片
# plt.figure(500)
# x_col = xcol  #数组逆序
# y_col = np.transpose(Data_Msc)
# plt.plot(x_col, y_col)
# plt.xlabel("Wavenumber(nm)")
# plt.ylabel("Absorbance")
# plt.title("The spectrum of the MSC for dataset",fontweight= "semibold",fontsize='large') #记得改名字MSC
# plt.show()
#
# #保存预处理后的数据
# np.savetxt(datareprocessing_path, Data_Msc, delimiter=',')
