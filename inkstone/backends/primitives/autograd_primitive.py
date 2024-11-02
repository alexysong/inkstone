from functools import partial
import scipy as sp

import autograd.numpy as anp
from autograd.extend import primitive, defvjp


# SETUP ################################################################################################################################################################################################
# Setup functions taken from https://github.com/HIPS/autograd/blob/master/autograd/numpy/linalg.py
_dot = partial(anp.einsum, '...ij,...jk->...ik')

# batched diag
_diag = lambda a: anp.eye(a.shape[-1]) * a


# batched diagonal, similar to matrix_diag in tensorflow
def _matrix_diag(a):
    reps = anp.array(a.shape)
    # reps[:-1] = 1
    # reps[-1] = a.shape[-1]
    reps = reps.at[slice(None, -1)].set(1)
    reps = reps.at[-1].set(a.shape[-1])
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(anp.tile(a, reps).reshape(newshape))

@primitive
def j1(x):
    return anp.array(sp.special.j1(x)), (x,)


def j1_bwd(res, g):
    """
    The n-th derivative of the Bessel function of the first kind of order v at x is: sp.special.jvp(v,x,n)
    """
    x = res[0]
    vjp = g * anp.array(sp.special.jvp(1, x, n=1))
    return (vjp,)


defvjp(j1, j1_bwd)



@primitive
def j0(x):
    """
    Bessel function of the first kind of order 0.
    """
    return anp.array(sp.special.j0(x)), (x,)


def j0_bwd(res, g):
    """
    The n-th derivative of the Bessel function of the first kind of order v at x is: sp.special.jvp(v,x,n)
    """
    x = res[0]
    vjp = g * anp.array(sp.special.jvp(0, x, n=1))
    return (vjp,)


defvjp(j0, j0_bwd)

@primitive
def eig(A):
    """
    Calculates the eigenvalues and eigenvectors of a square matrix A
    """
    return anp.linalg.eig(A)


def eig_bwd(res, g):
    """
    VJP taken from https://github.com/HIPS/autograd/blob/master/autograd/numpy/linalg.py
    """
    A = res[0]
    e, u = res[1]  # eigenvalues as 1d array, eigenvectors in columns
    n = e.shape[-1]
    ut = anp.swapaxes(u, -1, -2)

    ge, gu = g
    ge = _matrix_diag(ge)

    f = 1 / (e[..., anp.newaxis, :] - e[..., :, anp.newaxis] + 1.e-20)
    f -= _diag(f)

    r1 = f * _dot(ut, gu)
    r2 = -f * (_dot(_dot(ut, anp.conj(u)), anp.real(_dot(ut, gu)) * anp.eye(n)))
    vjp = _dot(_dot(anp.linalg.inv(ut), ge + r1 + r2), ut)

    if not anp.iscomplexobj(A):
        vjp = anp.real(vjp)
        # the derivative is still complex for real input (imaginary delta is allowed), real output
        # but the derivative should be real in real input case when imaginary delta is forbidden

    return (vjp,)


defvjp(eig, eig_bwd)