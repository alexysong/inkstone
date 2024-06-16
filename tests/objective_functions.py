import numpy as np
import autograd.numpy as npa

import torch

def eigen(H):
    """Objective function to return product of eigenvalues of H + product of eigenvector components"""
    w,v = npa.linalg.eig(H)
    return npa.prod(w) + npa.prod(npa.abs(v))

def scaled_sum(x):
    return np.pi*npa.sum(x)

def scaled_sum_torch(x):
    return np.pi*torch.sum(x)

def scaled_sum_torch_np(x):
    return np.float64(np.pi)*torch.sum(x)

def dot_product(x,y):
    return x.dot(y)

def reshape_2x3_to_3x2(H):
    H_3x2 = H.reshape((3,2))
    return H_3x2[0,1]*H_3x2[1,1]