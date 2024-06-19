import numpy as np
import scipy.linalg as sla
import autograd.numpy as anp

import torch

def eigen(H):
    """Objective function to return product of eigenvalues of H + product of eigenvector components"""
    w,v = anp.linalg.eig(H)
    return anp.prod(w) + anp.prod(anp.abs(v))

def scaled_sum(x):
    return np.pi*anp.sum(x)

def scaled_sum_torch(x):
    return np.pi*torch.sum(x)

def scaled_sum_torch_np(x):
    return np.float64(np.pi)*torch.sum(x)

def dot_product(x,y):
    return x.dot(y)

def reshape_2x3_to_3x2(A):
    A_3x2 = A.reshape((3,2))
    return A_3x2[0,1]*A_3x2[1,1]

def sum_of_lu_solve(A,b):
    lu, piv = sla.lu_factor(A)
    x = sla.lu_solve((lu, piv), b)
    return np.sum(x)