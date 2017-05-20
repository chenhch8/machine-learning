# -*- coding: utf-8 -*-

'''决策树模块'''

__author__ = 'chenhch8'

# 导入全局变量
from globalVar import get_value

from utils import *
import numpy as np
import queue

class DTree(object):
  def __init__(self, leaf_size = 50):
    # 叶子节点数
    self.leaf_size = leaf_size
    # 叶子节点值
    self.leaf = queue.Queue(maxsize=leaf_size)
    # 决策树
    self.tree = {}
    global train_data, train_class
    train_data = get_value('train_data')
    train_class = get_value('train_class')

  def __findBestBoundary(self, trainDataIndex, features):
    '''寻找最优分割——返回第几个样本，第几个特征'''
    max = None
    size = len(trainDataIndex)
    # [1] 随机抽取500个
    randomIndex = pitch(trainDataIndex, 500)
    # print(randomIndex)
    for feature in features:
      # 特征值去重
      _randomIndex = decrease(feature, randomIndex)
      # [2] 特征值排序
      quitSort(feature, 0, len(_randomIndex) - 1, _randomIndex)
      # [3] 寻找当前特征下的最优分割
      var, boundery = self.__bestBoundary(feature, _randomIndex)
      if max == None:
        max = var; ptr_boundery = boundery; ptr_feature = feature
      elif max < var:
        max = var; ptr_boundery = boundery; ptr_feature = feature
    # 返回第几个样本，第几个特征
    return ptr_boundery, ptr_feature

  def __bestBoundary(self, feature, randomIndex):
    '''寻找最优分割——最小化方差'''
    max = None; ptr = None
    size = len(randomIndex)
    # 遍历每个可能的分界线，注意，最后一个不会被遍历到
    for index, value in enumerate(randomIndex):
      # 左半均值 右半均值
      means = np.array([calcMean(randomIndex[:index]), calcMean(randomIndex[index:])])
      # 权重
      weights = np.array([index / size, (size - index) / size])
      # 加权平均方差
      var = sum(means * means * weights)
      if max == None:
        max = var; ptr = value
      elif max < var:
        max = var; ptr = value
    # 计算当分界线为最后一个元素时
    means = calcMean(randomIndex)
    var = means**2
    if max == None:
      max = var; ptr = randomIndex[-1]
    elif max < var:
      max = var; ptr = randomIndex[-1]
    return max, ptr

  def __setNLeaf(self, feature, index, _tree):
    '''设置非叶结点'''
    _tree['feature'] = feature
    _tree['value'] = train_data[feature][index]
    _tree['isLeaf'] = False
    _tree['less'] = {}
    _tree['greater'] = {}

  def __setLeaf(self, trainIndex, _tree):
    _tree['predict'] = calcMean(trainIndex)
    _tree['isLeaf'] = True

  def build(self, trainDataIndex, features):
    '''建造DTree'''
    self.leaf.put((trainDataIndex, features, self.tree))
    # 开始创建
    while self.leaf.qsize() < self.leaf_size and self.leaf.empty() == False:
      currDataIndex, currFeatures, currTree = self.leaf.get()
      # 寻找最优分裂点
      index, feature = self.__findBestBoundary(currDataIndex, currFeatures)
      # 找出分裂坐标
      ptr = quitSlice(feature,
                      0,
                      len(currDataIndex) - 1,
                      train_data[feature][index],
                      currDataIndex)
      # 去掉已选定的特征
      currFeatures.remove(feature)
      if ptr == 1: # 结点只有一个元素
        self.__setLeaf([currDataIndex[0]], currTree)
      elif ptr == len(currDataIndex) - 1 or ptr == 0: # 只有一个分支，则与父结点合并
        self.leaf.put((currDataIndex, currFeatures, currTree))
      else: # 有两个分枝
        leftDataIndex = currDataIndex[:ptr]
        rightDataIndex = currDataIndex[ptr:]
        # 设置非叶结点
        self.__setNLeaf(feature, index, currTree)
        self.leaf.put((leftDataIndex, currFeatures.copy(), currTree['less']))
        self.leaf.put((rightDataIndex, currFeatures, currTree['greater']))

    # 保存叶子结点
    while self.leaf.empty() != True:
      currDataIndex, currFeatures, currTree = self.leaf.get()
      self.__setLeaf(currDataIndex, currTree)

  def getTree(self):
    return self.tree

  def setTree(self, tree):
    self.tree = tree

  def predict(self, sample):
    '''对sample进行预测'''
    tree = self.tree
    try:
      while tree['isLeaf'] != True:
        if sample[tree['feature']] < tree['value']:
          tree = tree['less']
        else:
          tree = tree['greater']
      return tree['predict']
    except KeyError:
      print(tree)
      print('sample[tree[feature]]=%s; tree[value]=%s' % (sample[tree['feature']], tree['value']))
      raise KeyError