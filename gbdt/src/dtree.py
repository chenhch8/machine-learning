# -*- coding: utf-8 -*-

'''决策树模块'''

__author__ = 'chenhch8'

# 导入全局变量
from globalVar import get_value

from utils import *
import numpy as np
import queue

class DTree(object):
  def __init__(self, leaf_size = 0, learing_rate = 0):
    if leaf_size == 0: return
    # 叶子节点数
    self.leaf_size = leaf_size
    # 叶子节点值
    self.leaf = queue.Queue(maxsize=leaf_size)
    # 学习率
    self.learing_rate = learing_rate
    # 决策树
    self.tree = {}
    global train_data, train_class
    train_data = get_value('train_data')
    train_class = get_value('train_class')
    self.min_size = int(train_data.shape[1] * 0.001)
    self.max_size = int(train_data.shape[1] * 0.005)

  def __findBestBoundary(self, trainDataIndex, features):
    '''寻找最优分割——返回第几个样本，第几个特征'''
    max = None
    size = len(trainDataIndex)
    # [1] 随机抽取500个
    randomIndex = pitch(trainDataIndex, 1000)
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
    print('max: %s  ptr_boundery: %s  ptr_feature: %s  len: %s' % (max, ptr_boundery, ptr_feature, len(features)))
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
    _tree['predict'] = calcMean(trainIndex) * self.learing_rate
    _tree['isLeaf'] = True
    # 更新估计值
    F = get_value('F')
    for index in trainIndex:
      F[index] += _tree['predict']


  def build(self, trainDataIndex, features):
    '''建造DTree'''
    count = 0
    self.leaf.put((trainDataIndex, features, self.tree))
    # 开始创建
    while self.leaf.qsize() < self.leaf_size and self.leaf.empty() == False:
      # print(self.leaf.qsize())
      currDataIndex, currFeatures, currTree = self.leaf.get()
      if len(currFeatures) == 1:
        count += 1
        self.__setLeaf(currDataIndex, currTree)
        continue
      # 寻找最优分裂点
      index, feature = self.__findBestBoundary(currDataIndex, currFeatures)
      # 分裂
      leftDataIndex, rightDataIndex = quitSlice(feature,
                                                train_data[feature][index],
                                                currDataIndex)
      # 删除选定特征
      currFeatures.remove(feature)
      ls, rs = len(leftDataIndex), len(rightDataIndex)
      # 设置叶子节点元素数量，防止过拟合
      if ls > self.min_size and ls <= self.max_size:
        count += 1
        self.__setLeaf(leftDataIndex, currTree)
      if rs > self.min_size and rs <= self.max_size:
        count += 1
        self.__setLeaf(rightDataIndex, currTree)

      if ls > self.max_size and rs > self.max_size:
        self.__setNLeaf(feature, index, currTree)
        self.leaf.put((leftDataIndex, currFeatures.copy(), currTree['less']))
        self.leaf.put((rightDataIndex, currFeatures, currTree['greater']))
      elif ls < self.min_size and rs > self.max_size:
        self.leaf.put((rightDataIndex, currFeatures, currTree))
      elif ls > self.max_size and rs < self.min_size:
        self.leaf.put((leftDataIndex, currFeatures, currTree))
      else:
        count += 1
        self.__setLeaf(currDataIndex, currTree)
    # 保存叶子结点
    while self.leaf.empty() != True:
      count += 1
      currDataIndex, currFeatures, currTree = self.leaf.get()
      self.__setLeaf(currDataIndex, currTree)
    print('叶子结点数：%s' % count)


  def getTree(self):
    return self.tree


  def setTree(self, tree):
    self.tree = tree


  def predict(self, sample):
    '''对sample进行预测'''
    tree = self.tree
    while tree['isLeaf'] != True:
      if sample[tree['feature']] < tree['value']:
        tree = tree['less']
      else:
        tree = tree['greater']
    return tree['predict']
