# -*- coding: utf-8 -*-

'''数据规范化'''
import pandas as pd

file = '../data/discretization_data.xls'
data = pd.read_excel(file)
data = data[u'肝气郁结证型系数'].copy()
k = 4

# 等宽离散化
d1 = pd.cut(data, k, labels = list(range(k)))

# 等频离散化
w = [1.0 * i / k for i in range(k + 1)]
w = data.describe(percentiles = w)[4:4+k+1]
d2 = pd.cut(data, w, labels = list(range(k)))

# 聚类离散化
from sklearn.cluster import KMeans
# n_clusters 簇个数；n_jobs 并行化数量
kmodel = KMeans(n_clusters = k, n_jobs = 4)
# 训练模型
kmodel.fit(data.values.reshape(len(data), 1))
# 输出聚类中心，且排序
c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
# 相邻 两项求中点，作为边界点
w = c.rolling(center=False, window=2).mean()[1:]
# 加上首末边界点
w = [0] + list(w[0]) + [data.max()]
d3 = pd.cut(data, w, labels = list(range(k)))

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


'''自定义作图函数用于显示聚类结果'''
# d: 分类结果；k: 分类个数
def cluster_plot(d, k):
    plt.figure(figsize = (8, 3))
    for j in range(0, k):
        # data[[true, false, ...]]：筛选出为true的数据
        plt.plot(data[d==j], [j for i in d[d==j]], 'o')
    # 设置纵坐标刻度
    plt.ylim(-0.5, k-0.5)
    return plt

cluster_plot(d1, k).show()
cluster_plot(d2, k).show()
cluster_plot(d3, k).show()