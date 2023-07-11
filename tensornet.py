import torch
import torch.nn.functional as F
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from utils import MSE_torch, RSE_torch

alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

class TensorNetwork(torch.nn.Module):
    def __init__(self, adj_matrix, printing=False, ensure_connect = True):
        # Attributes
        # We only really use the upper triangular part of adj_matrix 
        super().__init__()
        assert adj_matrix.shape[0] == adj_matrix.shape[1], 'adj_matrix must be a square matrix.'
        self.shape = adj_matrix.shape
        self.num_core = self.shape[0]
        if ensure_connect:
            # Compression is by edge, hence forward does not run properly if the graph are not connected.
            # To avoid that, we added some 1s to ensure it is the case
            for i in range(self.num_core-1):
                if adj_matrix[i][i+1] == 0: adj_matrix[i][i+1] = 1
        self.core_shape = [[] for _ in range(self.num_core)]
        self.output_shape = []
        self.output_order = []
        self.edges = []
        self.cores = []
        self.mode = 'ltr' # forward mode: ltr (left to right), rtl, random, greedy
        if printing: print('=======  Tensornetwork object created  =======')

        # Fill core_shape with shape of the core and the edges with edge of form [id1, idx1, id2, idx2]
        for i in range(self.num_core):
            for j in range(i+1, self.num_core):
                if adj_matrix[i][j] > 0:
                    self.edges.append([i, len(self.core_shape[i]), j, len(self.core_shape[j])])
                    self.core_shape[i].append(adj_matrix[i][j])
                    self.core_shape[j].append(adj_matrix[i][j])

        # Diag values for output dim
        count = 0
        for i in range(self.num_core):
            if adj_matrix[i][i] > 0:
                self.output_shape.append(adj_matrix[i][i])
                self.core_shape[i].append(adj_matrix[i][i])
                self.output_order.append([count])
                count += 1
            else: self.output_order.append([])

        # Fill the cores attributes with tensor cores
        for shape in self.core_shape:
            tensor = torch.rand(shape, requires_grad=True)
            self.cores.append(tensor)

        if printing: print('cores shape: ', self.core_shape)
        if printing: print('cores shape: ', [list(core.shape) for core in self.cores])
        if printing: print('output shape: ', self.output_shape)
        if printing: print('output order list: ', self.output_order)
        if printing: print('==============================================')

    def _merge1(self, core, d1, d2):
        # Compresses a tensor along two indices
        assert d1 != d2, 'Indices should be different'
        if d2 < d1: d1, d2 = d2, d1
        D = len(core.shape)
        str1 = alphabet[:d1]
        str2 = alphabet[d1+1:d2]
        str3 = alphabet[d2+1:D]
        rule = str1 + alphabet[d1] + str2 + alphabet[d1] + str3 + ' -> '\
            + str1 + str2 + str3 # ex: 'abcbe -> ace'
        # print('  (compressing rule: {})'.format(rule))
        return torch.einsum(rule, core)

    def _merge2(self, core1, core2, d1, d2):
        # Compresses two tensor along an edge
        D1 = len(core1.shape)
        D2 = len(core2.shape)
        str1 = alphabet[:d1]
        str2 = alphabet[d1+1:D1]
        str3 = alphabet[D1:D1+d2]
        str4 = alphabet[D1+d2+1:D1+D2]
        rule = str1 + alphabet[d1] + str2 + ','\
            + str3 + alphabet[d1] + str4 + ' -> '\
            + str1 + str2 + str3 + str4 # ex: 'abc, dbf -> acdf'
        # print('  (compressing rule: {})'.format(rule))
        return torch.einsum(rule, core1, core2)

    def _pre_calculation(self, edges, edge_idx, cores):
        # pre-computes an estimation of the compression operator cost
        # taken idea from TNGA and TNLS
        i1, d1, i2, _ = edges[edge_idx]
        return np.prod(cores[i1].shape)*np.prod(cores[i2].shape)//cores[i1].shape[d1]

    def _compress(self, edges, edge_idx, cores, output_order, verbose=False):
        # Compresses and calculates the tensor network along an edge
        # The edges left to compress are in edges[edge_idx+1:]
        # The newly calculated core is appended on the {cores} list
        i1, d1, i2, d2 = edges.pop(edge_idx)
        if verbose: print('  (compressing: {}, {} at {}, {})'.format(cores[i1].shape, cores[i2].shape, d1, d2))
        D1 = len(cores[i1].shape)
        inew = len(cores)
        output = None
        if i1 == i2:
            output = self._merge1(cores[i1], d1, d2)
            cores.append(output)
            # print('debug info: ', [list(i.shape) for i in cores], output_order, i1, i2)
            output_order.append(output_order[i1])
            for i in range(len(edges)):
                if edges[i][0] == i1:
                    edges[i][0] = inew
                    edges[i][1] -= (0 if edges[i][1] < min(d1, d2) else 2 if edges[i][1] > max(d1, d2) else 1)
                if edges[i][2] == i1:
                    edges[i][2] = inew
                    edges[i][3] -= (0 if edges[i][3] < min(d1, d2) else 2 if edges[i][3] > max(d1, d2) else 1)
        else:
            output = self._merge2(cores[i1], cores[i2], d1, d2)
            cores.append(output)
            # print('debug info: ', [list(i.shape) for i in cores], output_order, i1, i2)
            output_order.append(output_order[i1]+output_order[i2])
            for i in range(len(edges)):
                if edges[i][0] == i1:
                    edges[i][0] = inew
                    edges[i][1] -= (0 if edges[i][1] < d1 else 1)
                if edges[i][0] == i2:
                    edges[i][0] = inew
                    edges[i][1] += (D1-1 if edges[i][1] < d2 else D1-2)
                if edges[i][2] == i1:
                    edges[i][2] = inew
                    edges[i][3] -= (0 if edges[i][3] < d1 else 1)
                if edges[i][2] == i2:
                    edges[i][2] = inew
                    edges[i][3] += (D1-1 if edges[i][3] < d2 else D1-2)
        if verbose: print('  (compressing done: {})'.format(output.shape))

    def compression_ratio(self):
        return (np.sum([np.prod(shape) for shape in self.core_shape]))/np.prod(self.output_shape)

    def reset(self):
        # Randomly reinitializes the weights of the tensor cores
        self.cores.clear()
        for shape in self.core_shape:
            tensor = torch.rand(shape, requires_grad=True)
            self.cores.append(tensor)

    def set_mode(self, mode):
        assert mode in ['ltr', 'rtl', 'random', 'greedy'], 'Invalid forward mode'
        self.mode = mode
        
    def forward(self, verbose=False):
        # Compresses and calculates the tensor network.
        edges = [[i for i in edge] for edge in self.edges] # copying edges and output_order over
        for _ in range(len(edges)):
            idx = 0 if self.mode == 'ltr' else \
                -1 if self.mode == 'rtl' else \
                np.random.randint(len(edges)) if self.mode == 'random' else \
                np.argmin([self._pre_calculation(edges, j, self.cores) for j in range(len(edges))])
            self._compress(edges, idx, self.cores, self.output_order, verbose=verbose)
        output = self.cores[-1].permute(self.output_order[-1])
        self.cores = self.cores[:self.num_core] # clean up the generated tensors (appended at the end of the list)
        self.output_order = self.output_order[:self.num_core] # clean up the generated output_order list
        if verbose: print('  (forward output shape: ', output.shape, ')')
        return output

    def fit(self, target, loss='MSE', lr=0.01, iteration=50, verbose=False):
        print('----- Fitting with {} loss, lr {}, {} iteration ... -----'.format(loss, lr, iteration))
        # Fits the tensor network {repeat} time with loss {loss}
        if loss == 'MSE': criterion = F.mse_loss
        else: raise NotImplementedError
        optimizer = torch.optim.SGD(self.cores, lr = lr)
        error = 1e9
        output = None

        for j in range(iteration):
            optimizer.zero_grad()
            temp = self.forward()
            loss = criterion(temp, target)
            loss.backward()
            optimizer.step()
            if verbose: print(' Lost at iteration {}: {}'.format(j, loss.item()))

        with torch.no_grad():
            output = self.forward()
            error = criterion(output, target)

        print('----- Done! Output error: {} -----'.format(error))
        return output, error
    
    def compute_loss(self, target, loss='MSE'):
        if loss == 'MSE': return MSE_torch(self.forward(), target)
        if loss == 'RSE': return RSE_torch(self.forward(), target)
        raise NotImplementedError

    def draw_graph(self):
        # Draw the graph using information from the adjcency matrix
        pass

if __name__ == '__main__':

    adj_matrix = np.array([
        [10, 1, 1], 
        [0, 11, 1], 
        [0, 0, 12]
    ])
    # upper all equal 1 for an outer-product thingy
    # they should not all 0 otherwise the graph would not connect

    target = torch.rand((10, 11, 12), requires_grad = False)
    # target = torch.einsum('i, j, k -> ijk', torch.rand((10,), requires_grad = False), torch.rand((11,), requires_grad = False), torch.rand((12,), requires_grad = False))

    for a in [1, 2, 3, 4]:
        adj_matrix[np.triu_indices(3, 1)] = a
        attemps = []
        net = TensorNetwork(adj_matrix, printing=False)
        net.set_mode('greedy')
        print('Compression ratio (log): ', -np.log(net.compression_ratio()))
        for i in range(1, 6):
            net.reset()
            _, error = net.fit(target, 'MSE', 0.01, 1000)
            attemps.append(error)
        print('Errors for a={}: '.format(a), attemps)


