import numpy as np
import torch
import matplotlib.pyplot as plt
from tensornet import TensorNetwork
from utils import MSE, RSE, truncatedSVD

### === Lena experiment ===

target = torch.tensor(plt.imread('lena_gray.gif'), requires_grad = False, dtype=torch.float32)
# plt.imshow(plt.imread('lena_gray.gif'))
# plt.show()
# print('Confirm: ', target.shape, type(target))
target_ = target[:, :, 0].reshape((4, 4, 4, 4, 4, 4, 4, 4, 4))/256
# print('asdfd', target_.shape, target_[0, 0, 0, 0, 0, 0, 0])
adj_matrix = np.identity(9, dtype='int')*4
for i in range(8): adj_matrix[i][i+1] = 4
print(adj_matrix)

net = TensorNetwork(adj_matrix)
print('Compression ratio (log): ', -np.log(net.compression_ratio()))
attemps = []
net.set_mode('greedy')
for _ in range(5):
    net.reset()
    _, error = net.fit(target_, 'MSE', 0.0001, 1000)
    attemps.append(error)
print('Errors for TT: ', attemps)

net = TensorNetwork(adj_matrix)
print('Compression ratio (log): ', -np.log(net.compression_ratio()))
adj_matrix[0][-1] = 4
attemps = []
net.set_mode('greedy')
for _ in range(5):
    net.reset()
    _, error = net.fit(target_, 'MSE', 0.0001, 1000)
    attemps.append(error)
print('Errors for TR: ', attemps)
