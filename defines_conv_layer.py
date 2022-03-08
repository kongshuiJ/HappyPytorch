# 卷积神经网络的编写要用到nn.Conv2d
# 该API意为进行2D的函数卷积层计算

import torch
import torch.nn as nn

# 1代表每个kernel的channel是1，5代表kernel的数量，同时也是输出到下一层的channel数量
layer = nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=0)
out   = torch.rand(1, 1, 28, 28)  # 1张图片，1channel，28*28

print('out.shape', out.shape)
print('layer.weight.size', layer.weight.size())
print('layer.bias.size', layer.bias.size())
