# CenterNet_onnx_rknn_horizon_tensorRT
CenterNet 部署版本，便于移植不同平台（onnx、tensorRT、rknn、Horizon）。

**本来想做基于CenterNet的单目3D目标检测，觉得要先把centenet给捣鼓清楚，花了一点时间把centernet的后处理手撸了一遍。**

centernet_onnx：onnx模型、测试图像、测试结果、测试demo脚本

centernet_TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

centernet_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

centernet_horizon：地平线模型、测试（量化）图像、测试结果、转换测试脚本、测试量化后onnx模型脚本

# onnx 测试效果

![image](https://github.com/cqu20160901/CenterNet_onnx_rknn_horizon_tensorRT/blob/main/centernet_onnx/test_onnx_result.jpg)
