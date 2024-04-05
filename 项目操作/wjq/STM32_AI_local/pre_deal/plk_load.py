import pickle
#
# # 替换为你的 .pkl 文件路径
# file_path = 'C:\\programming\\PycharmProjects\\STM32_AI\\tiny_training\\tinytraining\\algorithm\\assets\\mcu_models\\mbv2-w0.35.pkl'
#
# # 以二进制读取模式打开文件，并使用 pickle 加载
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
#
# # 现在，data 变量包含了 .pkl 文件中的原始对象，你可以根据需要对其进行操作
# print(data,len(data),type(data))


# import numpy as np
#
# meta_file = 'C:\\programming\\PycharmProjects\\STM32_AI\\tiny_training\\tinytraining\\algorithm\\assets\\mcu_models\\mbv2-w0.35.pkl'
# data = np.load(meta_file, allow_pickle=True)
# print(data)



import torch
import matplotlib.pyplot as plt
import json
f = open('C:\\programming\\PycharmProjects\\STM32_AI\\tiny_training\\tinytraining\\algorithm\\assets\\mcu_models\\mbv2-w0.35.pkl','rb')
data = torch.load(f,map_location='gpu')#可使用cpu或gpu
# print(data,len(data),type(data))
keys = data.keys()
print(keys,type(keys))