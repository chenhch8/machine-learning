# -*- coding: utf-8 -*-

#菜品盈利数据 帕累托图

import pandas as pd

file = '../data/catering_dish_profit.xls'
data = pd.read_excel(file, index_col = u'菜品名')
data = data[u'盈利'].copy()
# 降序排序
data.sort_values(ascending = False)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure()
# 将data以垂直条形图呈现
data.plot(kind = 'bar')
plt.ylabel(u'盈利（元）')
# data.cumsum()返回一个累积和向量，即此处的p为一个向量，其每个元素均为累积和比值
p = 1.0 * data.cumsum() / data.sum()
# 将p以默认形状——线条样式呈现
p.plot(color = 'r', secondary_y = True, style = '-o', linewidth = 2)
# 指定箭头的位置、样式和注释
# 参数：呈现的值，值的位置坐标，注解的位置坐标，箭头属性
# format(1, '.2%'): 返回一个str类型的值，该值为'100.00'
plt.annotate(format(p[6],'.4%'), xy = (6, p[6]), xytext = (6*0.9, p[6]*0.9), 
             arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.ylabel(u'盈利（比例）')
plt.show()