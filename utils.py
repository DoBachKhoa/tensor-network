import numpy as np
import torch

def RSE(y_true, y_pred):
    y_mean = torch.mean(y_true)
    num = torch.sum(torch.square(y_true-y_pred))
    den = torch.sum(torch.square(y_true-y_mean))
    return num/den

def MSE(y_true, y_pred):
    return torch.mean(torch.square(y_true-y_pred))

def truncatedSVD(C, delta):
    U, S, V = np.linalg.svd(C, full_matrices=True)
    eigenvalues = S.diagonal()
    temp = 0
    k = 0
    while temp + delta < 1:
        temp += eigenvalues[k]
        k += 1
    return U[:, :k], S[:k, :k], V[:k, :], k

def TT_SVD(A, eps):
    # Implements the TT-SVD algorithm (numbered 1) in the Tensor Train Decomposition paper
    rs = [1]
    ns = list(A.shape)
    d = len(ns)
    delta = np.linalg.norm(A)*eps/np.sqrt(d-1)
    C = A.copy()
    cores = []

    for k in range(1, d-1):
        C = np.reshape(C, (rs[k-1]*ns[k], -1))
        U, S, V, r = truncatedSVD(C, delta)
        rs.append(r)
        cores.append(np.reshape(U, (rs[k-1], ns[k], rs[k])))
        C = S @ (V.T)
    cores.append(C)
    rs.append(1)
    return cores, rs