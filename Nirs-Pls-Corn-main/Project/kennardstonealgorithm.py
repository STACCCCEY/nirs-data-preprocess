# select samples using Kennard-Stone algorithm
import numpy as np

# 一、Kennard-Stone算法原理
#
# Kennard-Stone算法原理：把所有的样本都看作训练集候选样本,依次从中挑选样本进训练集。首先选择欧氏距离最远的两个样本进入训练集，其后通过计算剩下的每一个样品到训练集内每一个已知样品的欧式距离，找到距已选样本最远以及最近的两个样本，并将这两个样本选入训练集，重复上述步骤直到样本数量达到要求。
#
# 二、Kennard-Stone算法作用
#
# Kennard-Stone算法作用：用于数据集的划分，使用算法，将输入的数据集划分为训练集、测试集，并同时输出训练集和测试集在原样本集中的编号信息，方便样本的查找。
#
# kennardstonealgorithm函数中，输入x_variables为样本集（二维numpy），k为挑选的train训练集的个数；输出为划分的训练集和测试集在原样本中的编号数组。
# --- input ---
# X : dataset of X-variables (samples x variables)
# k : number of samples to be selected
#
# --- output ---
# selected_sample_numbers : selected sample numbers (training data)
# remaining_sample_numbers : remaining sample numbers (test data)

def kennardstonealgorithm(x_variables, k):
    x_variables = np.array(x_variables)
    original_x = x_variables
    distance_to_average = ((x_variables - np.tile(x_variables.mean(axis=0), (x_variables.shape[0], 1))) ** 2).sum(axis=1)
    max_distance_sample_number = np.where(distance_to_average == np.max(distance_to_average))
    max_distance_sample_number = max_distance_sample_number[0][0]
    selected_sample_numbers = list()
    selected_sample_numbers.append(max_distance_sample_number)
    remaining_sample_numbers = np.arange(0, x_variables.shape[0], 1)
    x_variables = np.delete(x_variables, selected_sample_numbers, 0)
    remaining_sample_numbers = np.delete(remaining_sample_numbers, selected_sample_numbers, 0)
    for iteration in range(1, k):
        selected_samples = original_x[selected_sample_numbers, :]
        min_distance_to_selected_samples = list()
        for min_distance_calculation_number in range(0, x_variables.shape[0]):
            distance_to_selected_samples = ((selected_samples - np.tile(x_variables[min_distance_calculation_number, :],
                                                                        (selected_samples.shape[0], 1))) ** 2).sum(axis=1)
            min_distance_to_selected_samples.append(np.min(distance_to_selected_samples))
        max_distance_sample_number = np.where(
            min_distance_to_selected_samples == np.max(min_distance_to_selected_samples))
        max_distance_sample_number = max_distance_sample_number[0][0]
        selected_sample_numbers.append(remaining_sample_numbers[max_distance_sample_number])
        x_variables = np.delete(x_variables, max_distance_sample_number, 0)
        remaining_sample_numbers = np.delete(remaining_sample_numbers, max_distance_sample_number, 0)

    return selected_sample_numbers, remaining_sample_numbers