# -*- coding: utf-8 -*-

'''决策树模块'''

__author__ = 'chenhch8'

# 导入全局变量
from globalVar import *

from utils import *
import numpy as np
import queue
# 多进程
import os
import time
from multiprocessing import sharedctypes as sct
import ctypes


def initile(sh_train_data_, sh_shape_, sh_rdl_):
  global sh_train_data, sh_shape, sh_rdl
  sh_train_data, sh_shape, sh_rdl = sh_train_data_, sh_shape_, sh_rdl_


# 设置共享变量
def setShareMem():
  train_data, residual = get_value('train_data'),  get_value('residual')
  sh_shape = sct.RawArray(ctypes.c_int, train_data.shape)
  sh_train_data = sct.RawArray(ctypes.c_double, train_data.flat)
  sh_rdl = sct.RawArray(ctypes.c_double, residual)
  set_value('residual', sh_rdl)
  return sh_train_data, sh_shape, sh_rdl


def bestBoundary(feature, randomIndex):
  '''寻找最优分割——最小化方差'''
  max, ptr = None, None
  size = len(randomIndex)
  # 遍历每个可能的分界线，注意，最后一个不会被遍历到
  for index, value in enumerate(randomIndex):
    # 左半均值 右半均值
    means = np.array([calcMean(randomIndex[:index], sh_rdl), calcMean(randomIndex[index:], sh_rdl)])
    # 权重
    weights = np.array([index / size, (size - index) / size])
    # 加权平均方差
    var = sum(means * means * weights)
    if max == None:
      max = var; ptr = value
    elif max < var:
      max = var; ptr = value
  # 计算当分界线为最后一个元素时
  means = calcMean(randomIndex, sh_rdl)
  var = means**2
  if max == None:
    max = var; ptr = randomIndex[-1]
  elif max < var:
    max = var; ptr = randomIndex[-1]
  return max, ptr

def findBestBoundaryHelper(randomIndex, features):
  max = None
  for feature in features:
    # 特征值去重
    _randomIndex = decrease(feature, randomIndex, sh_train_data, sh_shape)
    # [2] 特征值排序
    quitSort(feature, 0, len(_randomIndex) - 1, _randomIndex, sh_train_data, sh_shape)
    # [3] 寻找当前特征下的最优分割
    var, boundery = bestBoundary(feature, _randomIndex)
    if max == None:
      max = var; ptr_boundery = boundery; ptr_feature = feature
    elif max < var:
      max = var; ptr_boundery = boundery; ptr_feature = feature
  # 返回第几个样本，第几个特征
  return max, ptr_boundery, ptr_feature


class DTree(object):
  def __init__(self, leaf_size, learing_rate):
    # 叶子节点数
    self.leaf_size = leaf_size
    # 叶子节点值
    self.leaf = queue.Queue(maxsize=leaf_size)
    # 学习率
    self.learing_rate = learing_rate
    # 决策树
    self.tree = {}
    global train_data, train_class, residual
    train_data, train_class, residual = get_value('train_data'), get_value('train_class'), get_value('residual')
    self.min_size = int(train_data.shape[1] * 0.001)
    self.max_size = int(train_data.shape[1] * 0.005)
    global pool_size
    pool_size = get_value('pool_size')


  def __findBestBoundary(self, trainDataIndex, features):
    '''寻找最优分割——返回第几个样本，第几个特征'''
    max = None
    size = len(trainDataIndex)
    # [1] 随机抽取1000个
    randomIndex = pitch(trainDataIndex, 1000)
    # 获取进程池
    pool = get_value('Pool')
    result = []
    # 并行化计算最优分割
    if len(features) >= 12:
      size = int(len(features) / pool_size)
      for i in range(pool_size):
        if i < pool_size - 1:
          result.append(pool.apply_async(findBestBoundaryHelper, (randomIndex, features[i*size:(i+1)*size])))
        else:
          result.append(pool.apply_async(findBestBoundaryHelper, (randomIndex, features[i*size:])))
    else:
      result.append(pool.apply_async(findBestBoundaryHelper, (randomIndex, features)))

    max = None
    for item in result:
      var, boundery, feature = item.get()
      if max == None:
        max, ptr_boundery, ptr_feature = var, boundery, feature
      elif max < var:
        max, ptr_boundery, ptr_feature = var, boundery, feature
    return ptr_boundery, ptr_feature


  def __setNLeaf(self, feature, index, _tree):
    '''设置非叶结点'''
    _tree['feature'] = feature
    _tree['value'] = train_data[feature][index]
    _tree['isLeaf'] = False
    _tree['less'] = {}
    _tree['greater'] = {}


  def __setLeaf(self, trainIndex, _tree):
    _tree['predict'] = calcMean(trainIndex, residual) * self.learing_rate
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
