# -*- coding = utf-8 -*-
# developer：HykQAQ
# development time：2022/2/23  18:45
# file :modelscore.py

import numpy as np
from sklearn.metrics import *

def evaluating(y_true, y_pre, samplesets="验证集"):
    """
    :param y_true: (n_samples, )
    :param y_pre: (n_samples, )
    :samplesets: string
    :return: None
    """
    evs_ = explained_variance_score(y_true, y_pre)
    mae_ = mean_absolute_error(y_true, y_pre)
    mse_ = mean_squared_error(y_true, y_pre)
    r2_ = r2_score(y_true, y_pre)
    rmse_ = np.sqrt(mse_)
    rpd_ = np.std(y_true)/rmse_

    print("*"*100)
    print(samplesets + ' 解释方差得分  平均绝对误差  决定系数  均方误差  均方根误差  相对分析误差')
    print('结果     %6.4f       %6.4f    %6.4f   %6.4f   %6.4f        %6.4f' % (evs_, mae_, r2_, mse_, rmse_, rpd_))
    print("*"*100)

np.random.seed(666)
y_true = np.random.randint(10, 100, 50)
noise = np.random.normal(size=(50,))
y_pre = [elem+n for elem, n in zip(list(y_true), list(noise))]
evaluating(y_true, y_pre)
