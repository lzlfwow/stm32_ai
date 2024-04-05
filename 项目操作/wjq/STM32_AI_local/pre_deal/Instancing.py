import pickle
with open('C:\\programming\\PycharmProjects\\STM32_AI\\tiny_training\\tinytraining\\algorithm\\assets\\mcu_models\\mbv2-w0.35.pkl', 'rb') as f:
 data = pickle.load(f)
 print(data,type(data))