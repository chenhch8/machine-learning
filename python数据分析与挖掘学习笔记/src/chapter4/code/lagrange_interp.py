# -*- coding: utf-8 -*-

'''
拉格朗日插值代码
'''

import pandas as pd
from scipy.interpolate import lagrange

file = '../data/catering_sale.xls'
save = '../tmp/sales.xls'

data = pd.read_excel(file)
# 过滤异常值，将其变为空值
data[u'销量'][(data[u'销量'] < 400) | (data[u'销量'] > 5000)] = None
# data.__getitem__(u'销量').__setitem__((data[u'销量'] < 400) | (data[u'销量'] > 5000), None)

# 插值函数
# s为列向量，n为被插值的位置，k为取前后的数据个数
def ployinterp_column(s, n, k = 5):
  # 取值，通过下标索引(s类型为np.array)
  y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
  # 剔除空值
  y = y[y.notnull()]
  # 插值并返回插值结果
  # 参数：x, y
  return lagrange(y.index, list(y))[n]

# 逐个元素判断是否需要插值
# print('data', data)
print('data.columns=%s' % data.columns)
print('len(data)=%s' % len(data))
for i in data.columns:
  for j in range(len(data)):
    # isnull 返回一个数组
    if (data[i].isnull())[j]:
      data[i][j] = ployinterp_column(data[i], j)
      # data.__getitem__(i).__setitem__(j, ployinterp_column(data[i], j))

data.to_excel(save, index=False, encoding='utf8')