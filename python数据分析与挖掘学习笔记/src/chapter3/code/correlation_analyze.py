# -*- coding: utf-8 -*-

'''
餐饮销量数据相关性分析
'''

import pandas as pd

file = '../data/catering_sale_all.xls'
data = pd.read_excel(file, index_col = u'日期')

# 相关系数矩阵
corr = data.corr(method='spearman')
# 只显示“百合酱蒸凤爪”与其它样式的相关系数
print('[1]')
print(corr[u'百合酱蒸凤爪'])
# 计算“百合酱蒸凤爪”与“翡翠蒸香茜饺”的相关系数
print('[2]')
print(corr[u'百合酱蒸凤爪'][u'翡翠蒸香茜饺'])
# 或者 print(data[u'百合酱蒸凤爪'].corr(data[u'翡翠蒸香茜饺'], method='spearman'))

# 显示相关系数矩阵图
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

plt.figure()
scatter_matrix(data, alpha=0.2, figsize=(12, 15), diagonal='hist')
plt.show()

''' ------- 输出 -------
[1]
百合酱蒸凤爪     1.000000
翡翠蒸香茜饺     0.009206
金银蒜汁蒸排骨    0.016799
乐膳真味鸡      0.455638
蜜汁焗餐包      0.098085
生炒菜心       0.308496
铁板酸菜豆腐     0.204898
香煎韭菜饺      0.127448
香煎罗卜糕     -0.090276
原汁原味菜心     0.428316
Name: 百合酱蒸凤爪, dtype: float64
[2]
0.00920580305184
'''