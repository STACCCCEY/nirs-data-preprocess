# -*- coding = utf-8 -*-
# developer：HykQAQ
# development time：2022/3/3  9:52
# file :shootoutProject.py

import datapreprocessing as dp
import pandas
import numpy as np
import matplotlib.pyplot as plt
# 155条药品数据
# 波长范围为 600nm - 1898nm，
# 间隔为 2 nm（700 个通道）

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data = pandas.read_csv("data/shootout/shootout.csv", header=None)
# 绘图
plt.figure(500)
# 通过索引进行读取数据
x_col = data.loc[0, :]
y_col = np.transpose(np.array(data.loc[1:156, :]))  # 数组逆置
plt.plot(x_col, y_col)
plt.show()

func = "CT"
Processeddata_MSC = dp.MMS(np.array(data.loc[1:156, :]))
plt.figure(500)
y_col = np.transpose(Processeddata_MSC)
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the {0} for corn dataset".format(func), fontweight="semibold", fontsize='x-large')
plt.savefig('.//Result//{0}.png'.format(func))
plt.show()

