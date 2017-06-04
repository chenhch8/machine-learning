# -*-coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---- plt.plot() ----
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)
plt.plot(x, y, 'b--')
plt.show()

# ---- plt.pie() ----
# 定义每一块的比例
sizes = [15, 30, 45, 10]
# 参数
opts = dict(
  # 定义标签
  labels = ('Frogs', 'Hogs', 'Dogs', 'Logs'),
  # 每一块颜色
  colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral'],
  # 显示突出
  explode = (0, 0.1, 0, 0)
)

plt.pie(sizes, **opts)
# 显示为圆(避免比例压缩为椭圆)
plt.axis('equal')
plt.show()

# ---- plt.hist() ----
# 生成1000个随机数
x = np.random.randn(1000)
# 分成10组进行绘制直方图
# 分组方法：
#  [1] 间距 = (max-min)/组数
#  [2] 根据间距和组数划分若区域
#  [3] 统计在各个区域中有集合x的元素的数量
plt.hist(x, 10)
plt.show()

# ---- DataFrame/Series.plot(kind = 'box') ----
# 1000个服从正态分布的随机数
x = np.random.randn(1000)
# 构造两列的DataFrame
D = pd.DataFrame([x, x+1]).T
# 调用Series内置的作图方法画图，用kind参数指定箱形图box
D.plot(kind='box')
plt.show()

# ---- DataFrame/Series.plot(logy = True) ----
x = pd.Series(np.exp(np.arange(20)))
x.plot(label = u'原始数据图', legend = True)
plt.show()
x.plot(logy = True, label = u'对数数据集', legend = True)
plt.show()