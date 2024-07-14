import numpy as np
from functools import partial
import scipy as sp

import jax.numpy as jnp
from jax import custom_vjp

"""
This file defines the custom vector-Jacobian products (vjp) to be used by an automatic differentiation (AD) library (autograd/jax/torch).
To add the vjp to a certain AD library, implement it manually using only differentiable functions from that AD library.

If there is a base scipy/numpy/other package function that you want to differentiate but is not natively included 
in the AD library, you can add it here.

You should not add a custom vjp here if, for your AD library of choice:
    1) The function already exists (e.g. numpy.sin is jax.numpy.sin in jax)
        Caveat: some functions (like jax.numpy.eig()) are natively included in the AD library, but not natively differentiable.
        However, there might be a differentiable alternative, like jax.numpy.eigvals().
    2) You've successfully differentiated the function using the AD library and found consistent results compared to 
        finite difference derivatives
    3) The function is your objective function. In this case, the function should be differentiated in your main script. 
        If the AD library is throwing errors when trying to differentiate your objective function, then you should look inside
        your objective for functions which are not natively differentiable and add them here with their custom vjp.

"""

# SETUP ################################################################################################################################################################################################
# Setup functions taken from https://github.com/HIPS/autograd/blob/master/autograd/numpy/linalg.py
_dot = partial(jnp.einsum, '...ij,...jk->...ik')

# batched diag
_diag = lambda a: jnp.eye(a.shape[-1]) * a


# batched diagonal, similar to matrix_diag in tensorflow
def _matrix_diag(a):
    reps = jnp.array(a.shape)
    # reps[:-1] = 1
    # reps[-1] = a.shape[-1]
    reps = reps.at[slice(None, -1)].set(1)
    reps = reps.at[-1].set(a.shape[-1])
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(jnp.tile(a, reps).reshape(newshape))


# JAX ################################################################################################################################################################################################
"""
To add a custom vjp for a function f(x) in jax:
    1) Define f(x) with the @custom_vjp jax decorator
        JAX treats f as a black-box function whose derivative (vjp) is supplied directly by our vjp definition. In other words,
        JAX won't look inside f(x) to trace its composite functions, instead obtaining the vjp directly from our given definition.
    2) Define the forward pass function f_fwd(x)
        Returns the result of the forward computation, f(x).
        You can add a second return value in the form of a tuple containing "residual" data. The 
        residual data (e.g. the input x) are intermediate results from the forward computation which can be used by the 
        backward computation f_bwd without needing to recalculate, since the forward pass happens anyway.
    3) Define the backward pass function f_bwd(res,g)
        The res parameter is the residual results tuple generated in (2). The g parameter is the gradient vector 
        (the "vector" in vector-Jacobian product); if your final objective function output is a scalar y, g = (dy/dx)^T 
        and has the same dimensions as x.
        Returns the vjp of f(x): matmul(g, df/dx).

        NOTE: from custom_vjp docs, the output of f_bwd must be a tuple of length equal to the number of arguments of the 
        primal function f(x). This is so the return tuple accounts for the fan-in of inputs to the function x. Hence, if f has 
        only 1 argument, the return is a tuple of length 1: (vjp,).

        NOTE: you have to be careful that the inputs x and outputs vjp to the function have the same type, but cannot use 
        type checking.
    4) Declare that your custom function has a forward and backward with f.defvjp(f_fwd,f_bwd)

"""


@custom_vjp
def j1(x):
    """
    Bessel function of the first kind of order 1.
    """
    return jnp.array(sp.special.j1(x))


def j1_fwd(x):
    return j1(x), (x,)


def j1_bwd(res, g):
    """
    The n-th derivative of the Bessel function of the first kind of order v at x is: sp.special.jvp(v,x,n)
    """
    x = res[0]
    vjp = g * jnp.array(sp.special.jvp(1, x, n=1))
    return (vjp,)


j1.defvjp(j1_fwd, j1_bwd)


@custom_vjp
def j0(x):
    return jnp.array(sp.special.j0(x))


def j0_fwd(x):
    """
    Bessel function of the first kind of order 0.
    """
    return j0(x), (x,)


def j0_bwd(res, g):
    """
    The n-th derivative of the Bessel function of the first kind of order v at x is: sp.special.jvp(v,x,n)
    """
    x = res[0]
    vjp = g * jnp.array(sp.special.jvp(0, x, n=1))
    return (vjp,)


j0.defvjp(j0_fwd, j0_bwd)


@custom_vjp
def eig(A):
    return jnp.linalg.eig(A)


def eig_fwd(A):
    """
    Calculates the eigenvalues and eigenvectors of a square matrix A
    """
    return eig(A), (A, eig(A))


def eig_bwd(res, g):
    """
    VJP taken from https://github.com/HIPS/autograd/blob/master/autograd/numpy/linalg.py
    """
    A = res[0]
    e, u = res[1]  # eigenvalues as 1d array, eigenvectors in columns
    n = e.shape[-1]
    ut = jnp.swapaxes(u, -1, -2)

    ge, gu = g
    ge = _matrix_diag(ge)

    f = 1 / (e[..., jnp.newaxis, :] - e[..., :, jnp.newaxis] + 1.e-20)
    f -= _diag(f)

    r1 = f * _dot(ut, gu)
    r2 = -f * (_dot(_dot(ut, jnp.conj(u)), jnp.real(_dot(ut, gu)) * jnp.eye(n)))
    vjp = _dot(_dot(jnp.linalg.inv(ut), ge + r1 + r2), ut)

    if not jnp.iscomplexobj(A):
        vjp = jnp.real(vjp)
        # the derivative is still complex for real input (imaginary delta is allowed), real output
        # but the derivative should be real in real input case when imaginary delta is forbidden

    return (vjp,)


eig.defvjp(eig_fwd, eig_bwd)