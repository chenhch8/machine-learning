# -*- coding: utf-8 -*-

import pandas as pd
file = '../data/catering_sale.xls'
data = pd.read_excel(file, index_col=u'日期')
result = data.describe()
print(result)


import matplotlib.pyplot as plt
# 以 dict 形式返回箱线图
p = data.boxplot(return_type='dict')
# 异常值保存在 fliers 中
x = p['fliers'][0].get_xdata()
y = p['fliers'][0].get_ydata()
y.sort()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
for i in range(len(x)):
  if i > 0:
    plt.annotate(y[i], xy=(x[i],y[i]), xytext=(x[i]+0.05-0.8/(y[i]-y[i-1]), y[i]))
  else:
    plt.annotate(y[i], xy=(x[i],y[i]), xytext=(x[i]+0.08, y[i]))
plt.show()