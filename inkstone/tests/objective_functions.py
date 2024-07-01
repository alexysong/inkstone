"""
Objective functions used in miscellaneous testing

"""

import numpy as np
import scipy as sp
import scipy.linalg as sla
import autograd.numpy as anp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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

def sinxsq(x):
    sum_of_squares = jnp.sum(jnp.power(x,2))
    sum_of_squares = sum_of_squares.tolist()
    return jnp.sin(sum_of_squares)    

def j1(x):
    return sp.special.jn(1,x)

if __name__ == '__main__':
    bessel_order = 0
    x = np.linspace(-5,5,100)
    y = jax.scipy.special.bessel_jn(x, v=0)
    x = np.linspace(-5,5,100).reshape(1,100)
    # y = jnp.real(jnp.power(1j,bessel_order)*jax.scipy.special.i0(-1j*x))
    print(y)
    plt.plot(x,y)
    plt.show()