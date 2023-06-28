import torch
import torch.nn.functional as F
import numpy as np
import math
import random
import matplotlib.pyplot as plt

alphabet = 'abcdefghijklmnopqrstuvwxyz'

class TensorNetwork(torch.nn.Module):
    def __init__(self, adj_matrix):
        # Attributes
        super().__init__()
        assert adj_matrix.shape[0] == adj_matrix.shape[1], 'adj_matrix must be a square matrix.'
        self.shape = adj_matrix.shape
        self.num_core = self.shape[0]
        self.core_shape = [[] for _ in range(self.num_core)]
        self.output_shape = []
        self.edges = []
        self.cores = []
        print('=======  Tensornetwork object created  =======')

        # Fill core_shape with shape of the core and the edges with edge of form [id1, idx1, id2, idx2]
        for i in range(self.num_core):
            for j in range(i+1, self.num_core):
                if adj_matrix[i][j] > 0:
                    self.edges.append([i, len(self.core_shape[i]), j, len(self.core_shape[j])])
                    self.core_shape[i].append(adj_matrix[i][j])
                    self.core_shape[j].append(adj_matrix[i][j])

        # Diag values for output dim
        for i in range(self.num_core):
            if adj_matrix[i][i] > 0:
                self.output_shape.append(adj_matrix[i][i])
                self.core_shape[i].append(adj_matrix[i][i])

        # Fill the cores attributes with tensor cores
        for shape in self.core_shape:
            tensor = torch.rand(shape, requires_grad=True)
            self.cores.append(tensor)

        print('cores shape: ', self.core_shape)
        print('core shape: ', [core.shape for core in self.cores])
        print('output shape: ', self.output_shape)
        print('==============================================')

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
        str2 = alphabet[d1+1:D1+d2]
        str3 = alphabet[D1:D1+d2]
        str4 = alphabet[D1+d2+1:D1+D2]
        rule = str1 + alphabet[d1] + str2 + ','\
            + str3 + alphabet[d1] + str4 + ' -> '\
            + str1 + str2 + str3 + str4 # ex: 'abc, dbf -> acdf'
        # print('  (compressing rule: {})'.format(rule))
        return torch.einsum(rule, core1, core2)

    def _compress(self, edges, edge_idx, cores, verbose=False):
        # Compresses and calculates the tensor network along an edge
        # The edges left to compress are in edges[edge_idx+1:]
        # The newly calculated core is appended on the {cores} list
        i1, d1, i2, d2 = edges[edge_idx]
        if verbose: print('  (compressing: {}, {} at {}, {})'.format(cores[i1].shape, cores[i2].shape, d1, d2))
        D1 = len(cores[i1].shape)
        inew = len(cores)
        output = None
        if i1 == i2:
            output = self._merge1(cores[i1], d1, d2)
            cores.append(output)
            for i in range(edge_idx+1, len(edges)):
                if edges[i][0] == i1:
                    edges[i][0] = inew
                    edges[i][1] -= (0 if edges[i][1] < min(d1, d2) else 2 if edges[i][1] > max(d1, d2) else 1)
                if edges[i][2] == i1:
                    edges[i][2] = inew
                    edges[i][3] -= (0 if edges[i][1] < min(d1, d2) else 2 if edges[i][1] > max(d1, d2) else 1)
        else:
            output = self._merge2(cores[i1], cores[i2], d1, d2)
            cores.append(output)
            for i in range(edge_idx+1, len(edges)):
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
                    edges[i][3] += (D1-1 if edges[i][1] < d2 else D1-2)
        if verbose: print('  (compressing done: {})'.format(output.shape))

    def forward(self, verbose=False):
        # Compresses and calculates the tensor network.
        edges = [[i for i in edge] for edge in self.edges]
        for i in range(len(edges)):
            self._compress(edges, i, self.cores, verbose=verbose)
        output = self.cores[-1]
        self.cores = self.cores[:self.num_core]
        if verbose: print('  (forward output shape: ', output.shape, ')')
        return output

    def fit(self, target, loss='MSE', lr=0.01, iteration=50, verbose=False):
        print('----- Fitting with {} loss, lr {}, {} iteration -----'.format(loss, lr, iteration))
        # Fits the tensor network {repeat} time with lost {loss}
        if loss == 'MSE': criterion = F.mse_loss
        optimizer = torch.optim.SGD(self.cores, lr = lr)
        error = 1e9
        output = None

        for j in range(iteration):
            optimizer.zero_grad()
            temp = self.forward(verbose=verbose)
            loss = criterion(temp, target)
            loss.backward()
            optimizer.step()
            if verbose: print(' Loss at iteration {}: {}'.format(j, loss.item()))

        with torch.no_grad():
            output = self.forward()
            error = criterion(output, target)

        print()
        print('Output error: {}'.format(error))
        return output, error

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
    net = TensorNetwork(adj_matrix)
    net.fit(target, 'MSE', 0.01, 100)


