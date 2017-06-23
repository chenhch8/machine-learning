# -*- coding: utf-8 -*-

'''文件路径配置文件'''

__author__ = 'chenhch8'

import os

config = {
    # 训练测试集文件
    'trainData': os.path.join('..', 'data', 'train_data_smaller.txt'),
    # 预测数据测试文件
    'targetData': os.path.join('..', 'data', 'test_data.txt'),
    # 训练模型文件
    'targetModel': os.path.join('..', 'output', 'gbdt_result.json'),
    # 预测预测结果
    'result': os.path.join('..', 'output', 'test_result.txt')
}