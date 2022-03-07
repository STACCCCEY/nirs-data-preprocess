# -*- coding = utf-8 -*-
# developer：HykQAQ
# development time：2022/3/2  19:27
# file :mangoProject.py

import pandas
import numpy as np
import matplotlib.pyplot as plt

# 11691条芒果数据
# 光谱范围 309nm - 1149nm
# 间隔为3nm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(500)
data = pandas.read_csv("data/mango/NAnderson2020MendeleyMangoNIRData.csv", header=None)
# 通过索引进行读取数据
x_col = data.loc[0, 17:297]
y_col = np.transpose(np.array(data.loc[1:, 17:297]))

plt.plot(x_col, y_col)
plt.show()
