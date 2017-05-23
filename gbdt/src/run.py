# -*- coding: utf-8 -*-

'''程序入口模块'''

__author__ = 'chenhch8'

from utils import loadTrainData
from gbdt import GDBT
import os

def StartTrain(filename, tree_size, leaf_size):
  loadTrainData(filename)
  myGdbt = GDBT(tree_size, leaf_size)
  # 开始训练
  myGdbt.buildGDBT()
  # myGdbt.predictTestData()

def StartPredict(filename, test_filename, saveName, tree_size, leaf_size):
  myGdbt = GDBT(tree_size, leaf_size)
  # 开始预测
  myGdbt.startPredict(filename, test_filename, saveName)

if __name__ == '__main__':
  choice = input('1. 训练模型； 2. 预测数据：')
  choice = int(choice)
  # 树数量 叶子数量
  tree_size, leaf_size = 100, 30
  if choice == 1:
    print('开始训练模型')
    filename = os.path.join('..', 'data', 'train_data_small.txt')
    # 开始训练
    StartTrain(filename, tree_size, leaf_size)
  elif choice == 2:
    print('开始进行预测')
    filename = os.path.join('..', 'output', 'gbdt_result.json')
    test_filename = os.path.join('..', 'data', 'test_data.txt')
    saveName = os.path.join('..', 'output', 'test_result.txt')
    # 开始预测
    StartPredict(filename, test_filename, saveName, tree_size, leaf_size)
  else:
    print('选择错误')