@[TOC](近红外光谱分析技术与应用--光谱预处理)

# 简介


<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">
<font size =4 color=bule >本文主要介绍光谱预处理技术对玉米以及小麦样本进行处理。
<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

## 1.玉米数据集
数据来源https://eigenvector.com/resources/data-sets/

 该数据集包含在 3 个不同的 NIR 光谱仪上测量的 80 个玉米样品，波长范围为 1100-2498nm

```python
data_path = './data/corn/m5.csv'  
data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
# 绘制原始图片
plt.figure(500)
x_col = np.linspace(1100, 2498, len(data[0, :]))  # 返回介于start和stop之间均匀间隔的num个数据的一维矩阵
y_col = np.transpose(data)  # 数组逆序
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the raw for corn dataset", fontweight="semibold", fontsize='x-large')
plt.savefig('.//Result//raw.png')
plt.show()

```




![绘制原始光谱](https://img-blog.csdnimg.cn/629420dd595544819c58a8fd5c64e77d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFfNTE0MjMyOTg=,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 调用预处理函数进行处理（以SNV为例）
```python
# 绘制预处理后原始图片
func = "SNV"
Processeddata = dp.SNV(data)

plt.figure(500)
x_col = np.linspace(1100, 2498, len(Processeddata[0, :]))
y_col = np.transpose(Processeddata)
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the {0} for corn dataset".format(func), fontweight="semibold", fontsize='x-large')
plt.savefig('.//Result//{0}.png'.format(func))
plt.show()

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/652c405cbdf64dcea22c594946f9a778.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFfNTE0MjMyOTg=,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)
## 2.小麦数据集
数据来源https://www.cNIRS.org/content.aspx?page_id=22&club_id=409746&module_id=239453
总共  250条小麦数据 光谱范围 730nm - 1100nm 间隔为0.5nm
```python
import pandas
import numpy as np
import matplotlib.pyplot as plt
import datapreprocessing as dp  # 导入预处理文件

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(500)
data = pandas.read_csv("data/ManufacturerA/Cal_ManufacturerA.csv", header=None)
# 通过索引进行读取数据
x_col = data.loc[0, 2:742]
y_col = np.transpose(np.array(data.loc[1:249, 2:742]))
plt.plot(x_col, y_col)
plt.show()
```
![小麦样本](https://img-blog.csdnimg.cn/5bafab9f229f461e9512f2cdfa99efe2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFfNTE0MjMyOTg=,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)
调用MSC对光谱进行处理后。
![MSC处理](https://img-blog.csdnimg.cn/1acd99e7339e4ae0822d1367587fc314.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFfNTE0MjMyOTg=,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

## 预处理函数

```python
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
```

代码来源https://blog.csdn.net/Echo_Code?type=blog，根据实际情况做出了具体调整。
