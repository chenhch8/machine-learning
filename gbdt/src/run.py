# -*- coding: utf-8 -*-

'''程序入口模块'''

__author__ = 'chenhch8'

from utils import loadTrainData
from gbdt import GDBT
from config import config

if __name__ == '__main__':
  # 树数量 叶子数量
  tree_size, leaf_size = 300, 30
  learing_rate = 0.1

  choice = input('1. 训练模型； 2. 预测数据：')
  choice = int(choice)

  if choice == 1:
    print('开始训练模型')
    # 装载数据
    loadTrainData(config['trainData'])
    myGdbt = GDBT(tree_size, leaf_size, learing_rate)
    # 开始训练
    myGdbt.buildGDBT(config['targetModel'])
  elif choice == 2:
    print('开始进行预测')
    myGdbt = GDBT(tree_size, leaf_size, learing_rate)
    # 开始预测
    myGdbt.startPredict(config['targetModel'], config['targetData'], config['result'])
  else:
    print('选择错误')