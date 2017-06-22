# -*- coding: utf-8 -*-

'''程序入口模块'''

__author__ = 'chenhch8'

from dtree import initile, setShareMem
from utils import loadTrainData
from multiprocessing import Pool
from globalVar import set_value
from gbdt import GDBT
import os


if __name__ == '__main__':
  # 树数量 叶子数量
  tree_size, leaf_size = 100, 20
  learing_rate = 0.1
  # 进程池中进程数量
  pool_size = 4

  choice = input('1. 训练模型； 2. 预测数据：')
  choice = int(choice)

  if choice == 1:
    print('开始训练模型')
    # 装载数据
    filename = os.path.join('..', 'data', 'train_data_smaller.txt')
    loadTrainData(filename)
    # 创建进程池
    pool = Pool(processes = pool_size, initializer = initile, initargs = setShareMem())
    # 保存为全局变量
    set_value('Pool', pool)
    set_value('pool_size', pool_size)
    # 创建 gbdt 实例
    myGdbt = GDBT(tree_size, leaf_size, learing_rate)
    # 开始训练
    myGdbt.buildGDBT()
    pool.close()
    pool.join()
  elif choice == 2:
    print('开始进行预测')
    filename = os.path.join('..', 'output', 'gbdt_result.json')
    test_filename = os.path.join('..', 'data', 'test_data.txt')
    saveName = os.path.join('..', 'output', 'test_result.txt')
    from gbdt import GDBT
    myGdbt = GDBT(tree_size, leaf_size, learing_rate)
    # 开始预测
    myGdbt.startPredict(filename, test_filename, saveName)
  else:
    print('输入错误，程序退出')