# -*- coding: utf-8 -*-

import numpy as np
import sys
import time
import json

'''工具模块'''

__author__ = 'chenhch8'

# 导入全局变量
from globalVar import *
init()
# set_value('SUM', 1866819)
# set_value('SUM', 100000)
set_value('SUM', 10000)
set_value('FEATURE', 201)

SUM = get_value('SUM')
FEATURE = get_value('FEATURE')

'''
装载训练集数据
'''
def loadTrainData(filename, pen = 0.9):
  '''装载训练数据并返回样本矩阵和对应的分类矩阵'''
  if pen >= 1 or pen <= 0:
    raise ValueError('比例范围错误，应在 0~1 之间')
  print('数据装载中...')
  # 设置训练数集和测试数集大小
  train_sum = int(SUM * pen)
  test_sum = SUM - train_sum
  start = time.time()
  with open(filename) as file:
    global train_data, train_class
    # 训练集特征矩阵
    train_data = np.zeros((train_sum, FEATURE), dtype=float)
    train_class = np.zeros(train_sum, dtype=float)
    # 测试集特征矩阵
    test_data = np.zeros((test_sum, FEATURE), dtype=float)
    test_class = np.zeros(test_sum, dtype=int)

    for index, value in enumerate(file.readlines()):
      value = value.split(' ')
      if index < train_sum:
        train_class[index] = value[0]
        value = list(map(lambda x: x.split(':'), value[1:]))
        for v in value:
          train_data[index][int(v[0])-1] = v[1]
      else:
        index -= train_sum
        test_class[index] = value[0]
        value = list(map(lambda x: x.split(':'), value[1:]))
        for v in value:
          test_data[index][int(v[0])-1] = v[1]

  # # 为后面的运行提高效率
  train_data = train_data.T

  # 保存进全局变量
  set_value('train_data', train_data)
  set_value('train_class', train_class)
  set_value('test_data', test_data)
  set_value('test_class', test_class)
  # 设置估计值
  set_value('F', np.zeros(train_class.shape, dtype=float))
  # 设置残差
  set_value('residual', np.zeros(train_class.shape, dtype=float))

  print('数据载入完成，耗时：%ss' % (time.time() - start))
  size = int(sys.getsizeof(train_data)) / 1024.0 / 1024.0 / 1024.0
  print('train_data矩阵大小:%sGb' % size)


'''
对train_data的第index行进行排序
'''
def quitSort(index, left, right, indexList, train_data, shape):
  if left >= right:
    return
  mid = np.random.randint(left, right+1)
  __swap(indexList, left, mid)
  i = left + 1; j = right
  # temp = train_data[index][indexList[left]]
  raw_sum = shape[1] * index
  temp = train_data[raw_sum + indexList[left]]
  while i <= j:
    # while i <= right and train_data[index][indexList[i]] < temp:
    #   i += 1
    # while j > left and train_data[index][indexList[j]] > temp:
    #   j -= 1
    while i <= right and train_data[raw_sum + indexList[i]] < temp:
      i += 1
    while j > left and train_data[raw_sum + indexList[j]] > temp:
      j -= 1
    if i <= j:
      __swap(indexList, i, j)
      i += 1; j -= 1
  if j >= left:
    __swap(indexList, left, j)
  quitSort(index, left, j - 1, indexList, train_data, shape)
  quitSort(index, j + 1, right, indexList, train_data, shape)

'''
找到k，同时将小于k的和大于k的划分到两边
'''
def quitSlice(index, k, indexList):
  global train_data
  obj = {'left': [], 'right': []}
  for i in indexList:
    if train_data[index][i] < k:
      obj['left'].append(i)
    else:
      obj['right'].append(i)
  return obj['left'], obj['right']


def __swap(indexList, i, j):
  temp = indexList[j]
  indexList[j] = indexList[i]
  indexList[i] = temp


'''
计算均值
'''
def calcMean(trainDataIndex, residual):
  sum = 0
  for i in trainDataIndex:
    sum += residual[i]
  return sum / len(residual)


'''
随机抽取n个元素
'''
def pitch(lists, n = 100):
  if len(lists) < n:
    return lists
  return np.random.choice(lists, n)


'''
去重
'''
from collections import defaultdict

def decrease(feature, lists, train_data, shape):
  # print('size:', hex(id(train_data)))
  sum_count = feature * shape[1]
  # [1] 去重
  obj = defaultdict(lambda: None)
  for i in lists:
    # if obj[train_data[feature][i]] == None:
    #   obj[train_data[feature][i]] = []
    # obj[train_data[feature][i]].append(i)
    if obj[train_data[sum_count + i]] == None:
      obj[train_data[sum_count + i]] = []
    obj[train_data[sum_count + i]].append(i)
  # [2] 抽取
  result = []
  for v in obj.values():
    result.append(np.random.choice(v, 1)[0])
  return result


'''
将字典保存到文件中
'''
def saveJson(filename, jsObj):
  with open(filename, 'w') as file:
    obj = json.dumps(jsObj)
    file.write(obj)

'''
将字典load进内存中
'''
def loadJson(filename):
  with open(filename, 'r') as file:
    obj = file.read()
  return json.loads(obj)
