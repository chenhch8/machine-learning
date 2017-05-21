# -*- coding: utf-8 -*-

'''GDBT模块'''

__author__ = 'chenhch8'

# 导入全局变量
from globalVar import get_value

from utils import saveJson
from utils import loadJson

from dtree import DTree
import time
import numpy as np
from collections import defaultdict

class GDBT(object):

  def __init__(self, tree_size, leaf_size):
    # 树数量
    self.tree_size = tree_size
    # 叶子数量
    self.leaf_size = leaf_size
    # 决策树
    self.dtrees = {}
    global train_data, train_class
    train_data = get_value('train_data')
    train_class = get_value('train_class')

  def __calcLoss(self, dtree):
    # 此处有问题，无法获得global train_data等，需手动装入，待调试
    train_data = get_value('train_data')
    train_class = get_value('train_class')
    
    train_data = train_data.T
    for index, data in enumerate(train_data):
      i = dtree.predict(data)
      train_class[index] -= i
      # print('train_data[%s]=%s;predict=%s' % (index, train_class[index], i))
    train_data = train_data.T
    mean = np.mean(abs(train_class))
    print('均方误差为：%s' % mean)
    return mean


  def buildGDBT(self):
    trainDataIndex = list(range(train_data.shape[1]))
    features = list(range(train_data.shape[0]))
    # print('dfhsfhsfhfhshf',train_data.shape[1])
    print('开始训练 %d 棵树，每颗树叶子结点为 %d' % (self.tree_size, self.leaf_size))
    start = time.time()
    for i in range(self.tree_size):
      dtree = DTree(self.leaf_size)
      print('训练第 #%d 棵树...' % (i + 1))
      # [1] 匹配最优残差决策树
      dtree.build(trainDataIndex, features.copy())
      # [2] 更新残差
      mean = self.__calcLoss(dtree)
      # [3] 将生成的决策树加入树集合中
      self.dtrees['tree_' + str(i)] = dtree.getTree()
      if mean < 0.05:
        break
    end = time.time()
    print('训练完成，用时 %ss' % (end - start))
    # [3] 保存决策树集合
    saveJson('../output/gbdt_result.json', self.dtrees)
    # [4] 测试样本测试
    self.__predictTestData()

  def __predictTestData(self):
    '''测试样本测试'''
    test_data, test_class = get_value('test_data'), get_value('test_class')
    dtree = DTree(self.leaf_size)
    count = 0
    for i in range(test_data.shape[0]):
      v = 0
      for k in self.dtrees:
        dtree.setTree(self.dtrees[k])
        v += dtree.predict(test_data[i])
      v = 1 if v >= 0.5 else 0
      # print('#%s 预测值： %s； 真实值： %s' % (i+1, v, test_class[i]))
      count += abs(v - test_class[i])
    print('分类错误数为：%s； 错误率：%s' % (count, count / test_data.shape[0]))

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