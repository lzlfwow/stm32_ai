
import torch.onnx

# 步骤1: 加载.pkl模型
model = torch.load('C:\\programming\\PycharmProjects\\STM32_AI\\tiny_training\\tinytraining\\algorithm\\assets\\mcu_models\\mbv2-w0.35.pkl',map_location='gpu')#可使用cpu或gpu
print(model,type(model))

model_save=torch.load('C:\\programming\\PycharmProjects\\STM32_AI\\pre_deal\\model_instance.pth',map_location='gpu')
print(model_save,type(model_save))
# torch.save(model, 'model_instance.pth')
# # 确保模型处于评估模式
# model.eval()
#
# # 步骤2: 准备一个输入张量，这里的尺寸需要根据你的模型来确定
# # 这里的例子假设模型接收3通道的224x224图片
# input_tensor = torch.randn(1, 3, 224, 224)
#
# # 步骤3: 模型转换
# # 设置输出文件名
# output_onnx_file = 'model.onnx'
# # 调用export函数进行转换
# torch.onnx.export(model,               # 待转换的模型
#                   input_tensor,        # 模型的输入，用于推断图形结构
#                   output_onnx_file,    # 输出文件名
#                   export_params=True,  # 是否导出训练后的参数
#                   opset_version=11,    # ONNX版本，根据需要选择
#                   do_constant_folding=True,  # 是否执行常量折叠优化
#                   input_names=['input'],     # 输入名
#                   output_names=['output'],   # 输出名
#                   dynamic_axes={'input': {0: 'batch_size'},  # 批量大小可变
#                                 'output': {0: 'batch_size'}})
#
#
# print(f"模型已转换为ONNX格式并保存为{output_onnx_file}")
