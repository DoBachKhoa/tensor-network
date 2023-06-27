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
        assert self.shape[0] == self.shape[1], 'adj_matrix must be a square matrix.'
        self.shape = adj_matrix.shape
        self.num_core = self.shape[0]
        self.core_shape = [[] for _ in range(self.num_core)]
        self.edges = []
        self.cores = []

        # Fill core_shape with shape of the core and the edges with edge of form [id1, idx1, id2, idx2]
        for i in range(self.num_core):
            for j in range(i-1):
                if adj_matrix[i][j] > 0:
                    self.edges.append([i, len(self.core_shape[i]), j, len(self.core_shape[j])])
                    self.core_shape[i].append(adj_matrix[i][j])
                    self.core_shape[j].append(adj_matrix[i][j])

        # Fill the cores attributes with tensor cores
        for shape in self.core_shape:
            tensor = torch.rand(shape, requires_grad=True)
            self.cores.append(tensor)

    def _merge1(self, core, d1, d2):
        # Compresses a tensor along two indices
        assert d1 != d2, 'Indices should be different'
        if d2 < d1: d1, d2 = d2, d1
        D = len(core.shape)
        str1 = alphabet[:d2]
        str2 = alphabet[d2+1:D]
        rule = str1 + alphabet[d1] + str2 + ' -> ' + str1 + str2 # ex: 'abcbe -> abce'
        return torch.einsum(rule, core)

    def _merge2(self, core1, core2, d1, d2):
        # Compresses two tensor along an edge
        D1 = len(core1.shape)
        D2 = len(core2.shape)
        str1 = alphabet[:D1]
        str2 = alphabet[D1:D1+d2]
        str3 = alphabet[D1+d2+1:D1+D2]
        rule = str1 + ',' + str2 + alphabet[d1] + str3 + ' -> ' + str1+str2+str3 # ex: 'abc, dbf -> abcdf'
        return torch.einsum(rule, core1, core2)

    def _compress(self, edges, edge_idx, cores):
        # Compresses and calculates the tensor network along an edge
        # The edges left to compress are in edges[edge_idx+1:]
        # The newly calculated core is appended on the {cores} list
        i1, d1, i2, d2 = edges[edge_idx]
        D1 = len(cores[i1].shape)
        inew = len(cores)
        if i1 == i2:
            cores.append(self._merge1(cores[i1], d1, d2))
            for i in range(edge_idx+1, len(edges)):
                if edges[i][0] == i1:
                    edges[i][0] = inew
                    edges[i][1] -= (0 if edges[i][1] < max(d1, d2) else 1)
                if edges[i][2] == i1:
                    edges[i][2] = inew
                    edges[i][3] -= (0 if edges[i][3] < max(d1, d2) else 1)
        else:
            cores.append(self._merge2(cores[i1], cores[i2], d1, d2))
            for i in range(edge_idx+1, len(edges)):
                if edges[i][0] == i1: edges[i][0] = inew
                if edges[i][0] == i2:
                    edges[i][0] = inew
                    edges[i][1] += (D1 if edges[i][1] < d2 else D1-1)
                if edges[i][2] == i1: edges[i][2] = inew
                if edges[i][2] == i2:
                    edges[i][2] = inew
                    edges[i][3] += (D1 if edges[i][1] < d2 else D1-1)

    def forward(self):
        # Compresses and calculates the tensor network.
        edges = [[i for i in edge] for edge in self.edges]
        for i in range(len(edges)):
            self._compress(edges, i, self.cores)
        output = self.cores[-1]
        self.cores = self.cores[:self.num_core]
        return output

    def fit(self, target, repeat=1, loss='MSE', lr=0.01, iteration=50):
        # Fits the tensor network {repeat} time with lost {loss}
        if loss == 'MSE': criterion = F.mse_loss()
        optimizer = torch.optim.SGD(learning_rate = lr)
        loss = 1e9
        output = None
        for _ in range(repeat):
            pass
        return output, loss

    def draw_graph(self):
        # Draw the graph using information from the adjcency matrix
        pass



