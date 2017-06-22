# -*- coding: utf-8 -*-

'''GDBT模块'''

__author__ = 'chenhch8'

# 导入全局变量
from globalVar import get_value, set_value

from utils import saveJson, loadJson

from dtree import DTree
import time
import numpy as np
from collections import defaultdict


class GDBT(object):

  def __init__(self, tree_size, leaf_size, learning_rate):
    # 树数量
    self.tree_size = tree_size
    # 叶子数量
    self.leaf_size = leaf_size
    # 学习率
    self.learning_rate = learning_rate
    # 决策树
    self.dtrees = {}
    global train_data, train_class
    train_data = get_value('train_data')
    train_class = get_value('train_class')


  def __calcLoss(self):
    F, residual = get_value('F'), get_value('residual')
    for index in range(train_class.shape[0]):
      residual[index] = train_class[index] - F[index]
    _mean = np.mean(abs(residual))
    # print('均值残差：%s' % mean)
    return _mean


  def buildGDBT(self):
    trainDataIndex = list(range(train_data.shape[1]))
    features = list(range(train_data.shape[0]))
    # 设置估计值
    set_value('F', np.zeros(train_class.shape, dtype=float))
    # 设置残差
    set_value('residual', np.zeros(train_class.shape, dtype=float))

    print('开始训练 %d 棵树，每颗树叶子结点最多为 %d, 学习率为 %s' % (self.tree_size, self.leaf_size, self.learning_rate))
    dtree = DTree(self.leaf_size, self.learning_rate)
    start = time.time()
    for i in range(self.tree_size):
      print('训练第 #%d 棵树...' % (i + 1))
      # [1] 计算残差
      _mean = self.__calcLoss()
      # [2] 匹配最优残差决策树 + 更新估计值
      dtree.build(trainDataIndex, features.copy())
      print('均值残差：%s  累积耗时：%ss' % (_mean, time.time() - start))
      # [3] 将生成的决策树加入树集合中
      self.dtrees['tree_' + str(i)] = dtree.getTree()
      # [4] 清除tree，给下一轮迭代使用
      dtree.setTree({})
    end = time.time()
    print('训练完成，用时 %smin' % ((end - start) / 60.0))
    # [3] 保存决策树集合
    saveJson('../output/gbdt_result_tmp.json', self.dtrees)
    # [4] 测试样本测试
    self.predictTestData()


  def __predictHelper(self, my_data, my_class, name):
    dtree = DTree(self.leaf_size, self.learning_rate)
    count = 0
    start = time.time()
    for i in range(my_data.shape[0]):
      v = 0
      for k in self.dtrees:
        dtree.setTree(self.dtrees[k])
        v += dtree.predict(my_data[i])
      v = 1 if v >= 0.5 else 0
      count += abs(v - my_class[i])
    end = time.time()
    print('%s：' % name)
    print('\t分类错误数为：%s； 错误率：%s' % (count, count / my_data.shape[0]))
    pried = end - start
    print('\t总耗时：%ss； 平均耗时：%ss' % (pried, pried / my_class.shape[0]))


  def predictTestData(self):
    '''误差计算'''
    if len(self.dtrees) == 0:
      self.dtrees = loadJson('../output/gbdt_result.json')
    print('误差计算中...')
    self.__predictHelper(train_data.T, train_class, '训练集')

    test_data, test_class = get_value('test_data'), get_value('test_class')
    self.__predictHelper(test_data, test_class, '测试集')


  def startPredict(self, filename, test_filename, saveName):
    '''预测'''
    print('数据预测中...')
    self.dtrees = loadJson(filename)
    dtree = DTree()
    result = 'id,label\n'
    obj = np.zeros(201, dtype=float)
    with open(test_filename, 'r') as file:
      for item in file.readlines():
        obj.fill(0)
        temp = item.split(' ')
        id = temp[0]
        temp = map(lambda x: x.split(':'), temp[1:])
        for k in temp:
          obj[int(k[0])-1] = k[1]
        v = 0
        for key in self.dtrees:
          dtree.setTree(self.dtrees[key])
          v += dtree.predict(obj)
        v = 1 if v >= 0.5 else 0
        result += id + ',' + str(v) + '\n'
    with open(saveName, 'w') as f:
      f.write(result)
