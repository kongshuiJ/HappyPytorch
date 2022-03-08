import torch
import numpy as np

# 1. 直接生成张量
# 由原始数据直接生成张量, 张量类型由原始数据类型决定。
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print('x_data', x_data)

# 2. 通过Numpy数组来生成张量
# 由已有的Numpy数组来生成张量(反过来也可以由张量来生成Numpy数组)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print('x_np', x_np)

# 3. 通过已有的张量来生成新的张量
# 新的张量将继承已有张量的数据属性(结构、类型), 也可以重新指定新的数据类型。
x_ones = torch.ones_like(x_data)  # 保留 x_data 的属性
print('x_ones', x_ones)

# 重写 x_data 的数据类型int -> float
x_rand = torch.rand_like(x_data, dtype=torch.float)
print('x_rand', x_rand)

# 4. 通过指定数据维度来生成张量
# shape是元组类型, 用来描述张量的维数,
# 下面3个函数通过传入shape来指定生成张量的维数。
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 张量属性
tensor = torch.rand(3, 4, )
print(f"Shape of tensor: {tensor.shape}")  # 维数
print(f"Datatype of tensor: {tensor.dtype}")  # 数据类型
print(f"Device tensor is stored on: {tensor.device}")  # 存储设备

# 张量运算

# 判断当前环境GPU是否可用, 然后将tensor导入GPU内运行
# if torch.cuda.is_available():
#     print('gpu is available')
#     tensor = tensor.to('cuda')
# else:
#     print('gpu is not available')

# 张量的索引和切片
tensor = torch.ones(4, 4)
tensor[:, 1] = 0  # # 将第1列(从0开始)的数据全部赋值为0
print(tensor)

# 张量的拼接
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # 4, 16
print('t1', t1)

t1 = torch.cat([tensor, tensor, tensor], dim=0)  # 16, 4
print('t1', t1)

# 张量的乘积和矩阵乘法
# 逐个元素相乘结果
print('tensor.mul(tensor):', tensor.mul(tensor))
# 等价写法:
print('tensor * tensor', tensor * tensor)
# 张量与张量的矩阵乘法:
print('tensor.matmul(tensor.T)', tensor.matmul(tensor.T))
# 等价写法:
print('tensor @ tensor.T', tensor @ tensor)

# 自动赋值运算
print('tensor', tensor)
tensor.add_(5)
print('tensor', tensor)

# 由张量变换为Numpy array数组
t = torch.ones(5)
print('t', t)
n = t.numpy()
print('n', n)

# 修改张量的值，则Numpy array数组值也会随之改变。
t.add_(2)
print('t', t)
print('n', n)

# 由Numpy array数组转为张量
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print('t', t)
print('n', n)
