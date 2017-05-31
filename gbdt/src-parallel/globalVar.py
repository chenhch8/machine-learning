# -* coding: utf-8 -*-

'''全局共享变量'''

__author__ = 'chenhch8'

def init():
  global _global_dict
  _global_dict = {}

def set_value(key, value):
  """ 定义一个全局变量 """
  _global_dict[key] = value

def get_value(key):
  try:
    return _global_dict[key]
  except KeyError:
    return None