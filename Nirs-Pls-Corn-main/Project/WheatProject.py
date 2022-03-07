# -*- coding = utf-8 -*-
# developer：HykQAQ
# development time：2022/3/2  14:08
# file :WheatProject.py

import pandas
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import datapreprocessing as dp  # 导入预处理文件
# 250条小麦数据
# 光谱范围 730nm - 1100nm
# 间隔为0.5nm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(500)
data = pandas.read_csv("data/ManufacturerA/Cal_ManufacturerA.csv", header=None)
# 通过索引进行读取数据
x_col = data.loc[0, 2:742]
y_col = np.transpose(np.array(data.loc[1:249, 2:742]))
y_col = dp.MSC(y_col)
plt.plot(x_col, y_col)
plt.savefig('.//Result//MSCwheat.png')
plt.show()

