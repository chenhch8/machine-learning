# -*- coding: utf-8 -*-

import pandas as pd

file = '../data/catering_sale.xls'
# 读取数据，指定‘日期’列为索引列
data = pd.read_excel(file, index_col=u'日期')
# 过滤异常数据
data = data[(data[u'销量'] > 400) & (data[u'销量'] < 5000)]
# 保存基本统计量
statistics = data.describe()

# 极差
statistics.loc['极差'] = statistics.loc['max'] - statistics.loc['min']
# 变异系数
statistics.loc['变异系数'] = statistics.loc['std'] / statistics.loc['mean']
# 四分位数间距
statistics.loc['四分位数间距'] = statistics.loc['75%'] - statistics.loc['25%']

print(statistics)

'''
                 销量
count    195.000000
mean    2744.595385
std      424.739407
min      865.000000
25%     2460.600000
50%     2655.900000
75%     3023.200000
max     4065.200000
极差      3200.200000
变异系数       0.154755
四分位数间距   562.600000
'''