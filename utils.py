import numpy as np
import torch

def RSE_torch(y_true, y_pred):
    y_mean = torch.mean(y_true)
    num = torch.sum(torch.square(y_true-y_pred))
    den = torch.sum(torch.square(y_true-y_mean))
    return num/den

def MSE_torch(y_true, y_pred):
    return torch.mean(torch.square(y_true-y_pred))

def RSE(y_true, y_pred):
    y_mean = np.mean(y_true)
    num = np.sum(np.square(y_true-y_pred))
    den = np.sum(np.square(y_true-y_mean))
    return num/den

def MSE(y_true, y_pred):
    return np.mean(np.square(y_true-y_pred))

def truncatedSVD(C, delta):
    U, S, V = np.linalg.svd(C, full_matrices=False)
    temp = 0
    k = len(S)
    while temp < delta**2 and k > 0:
        temp += S[k-1]**2
        k -= 1
    k += 1
    return U[:, :k], S[:k], V[:k, :], k

def TT_SVD(A, eps):
    # Implements the TT-SVD algorithm (numbered 1) in the Tensor Train Decomposition paper
    rs = [1]
    ns = list(A.shape)
    d = len(ns)
    delta = np.linalg.norm(A)*eps/np.sqrt(d-1)
    C = A.copy()
    cores = []

    for k in range(1, d):
        C = np.reshape(C, (rs[k-1]*ns[k], -1))
        U, S, V, r = truncatedSVD(C, delta)
        rs.append(r)
        cores.append(np.reshape(U, (rs[k-1], ns[k], rs[k])))
        C = np.diag(S) @ (V)
    cores.append(C)
    cores[0] = np.reshape(cores[0], cores[0].shape[1:])
    return cores, rs

def TT_compress(cores):
    output = cores[0]
    for i in range(1, len(cores)-1):
        core = cores[i]
        output = np.einsum('...i, ijk->...jk', output, core)
    output = np.einsum('...i, ij->...j', output, cores[-1])
    return output

def TT_CR(cores, target):
    return np.sum([np.prod(core.shape) for core in cores])/np.prod(target.shape)

def TT_loss(train, target, loss='MSE'):
    if loss == 'MSE': return MSE(train, target)
    if loss == 'RSE': return RSE(train, target)
    raise NotImplementedError