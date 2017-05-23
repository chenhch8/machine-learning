# -*- coding: utf-8 -*-

'''程序入口模块'''

__author__ = 'chenhch8'

from utils import loadTrainData
from gbdt import GDBT
import os

if __name__ == '__main__':
  # 树数量 叶子数量
  tree_size, leaf_size = 100, 20

  choice = input('1. 训练模型； 2. 预测数据：')
  choice = int(choice)

  if choice == 1:
    print('开始训练模型')
    filename = os.path.join('..', 'data', 'train_data_small.txt')
    # 装载数据
    loadTrainData(filename)
    myGdbt = GDBT(tree_size, leaf_size)
    # 开始训练
    myGdbt.buildGDBT()
  elif choice == 2:
    print('开始进行预测')
    filename = os.path.join('..', 'output', 'gbdt_result.json')
    test_filename = os.path.join('..', 'data', 'test_data.txt')
    saveName = os.path.join('..', 'output', 'test_result.txt')
    myGdbt = GDBT(tree_size, leaf_size)
    # 开始预测
    myGdbt.startPredict(filename, test_filename, saveName)
  else:
    print('选择错误')