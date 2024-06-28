import numpy as np
from functools import partial
import scipy as sp

import jax.numpy as jnp
from jax import custom_vjp



# SETUP ################################################################################################################################################################################################
# Setup functions taken from https://github.com/HIPS/autograd/blob/master/autograd/numpy/linalg.py
_dot = partial(jnp.einsum, '...ij,...jk->...ik')

# batched diag
_diag = lambda a: jnp.eye(a.shape[-1])*a

# batched diagonal, similar to matrix_diag in tensorflow
def _matrix_diag(a):
    reps = jnp.array(a.shape)
    # reps[:-1] = 1
    # reps[-1] = a.shape[-1]
    reps = reps.at[slice(None,-1)].set(1)
    reps = reps.at[-1].set(a.shape[-1])
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(jnp.tile(a, reps).reshape(newshape))



# JAX ################################################################################################################################################################################################
@custom_vjp
def j1(x):
    return jnp.array(sp.special.j1(x))

def j1_fwd(x):
    """
    Second return value is tuple containing 'residual' data to be stored for use by back pass
    """
    return j1(x), (x,)

def j1_bwd(res, g):
    """
    Parameters
    ----------
    res: residuals from fwd func (tuple)
    g  : vector to backprop (scalar, for some nodes)

    Return
    ------
    vjp: vector-Jacobian product of g with Jacobian (tuple)

    NOTE: from custom_vjp docs, the output of bwd must be a tuple of length equal to the number 
    of arguments of the primal function. Here, sinxsq has only 1 argument, so create a tuple of 
    length 1 with (vjp,)

    NOTE: you have to be careful that the inputs x and outputs vjp to the function have the 
    same type, but cannot use type checking
    """
    x = res[0]
    vjp = g*jnp.array(sp.special.jvp(1, x, n=1))
    return (vjp,)

j1.defvjp(j1_fwd, j1_bwd)

@custom_vjp
def j0(x):
    return jnp.array(sp.special.j0(x))

def j0_fwd(x):
    """
    Second return value is tuple containing 'residual' data to be stored for use by back pass
    """
    return j0(x), (x,)

def j0_bwd(res, g):
    """
    Parameters
    ----------
    res: residuals from fwd func (tuple)
    g  : vector to backprop (scalar, for some nodes)

    Return
    ------
    vjp: vector-Jacobian product of g with Jacobian (tuple)

    NOTE: from custom_vjp docs, the output of bwd must be a tuple of length equal to the number 
    of arguments of the primal function. Here, sinxsq has only 1 argument, so create a tuple of 
    length 1 with (vjp,)

    NOTE: you have to be careful that the inputs x and outputs vjp to the function have the 
    same type, but cannot use type checking
    """
    x = res[0]
    vjp = g*jnp.array(sp.special.jvp(0, x, n=1))
    return (vjp,)

j0.defvjp(j0_fwd, j0_bwd)


@custom_vjp
def eig(A):
    return jnp.linalg.eig(A)

def eig_fwd(A):
    """
    Second return value is tuple containing 'residual' data to be stored for use by back pass
    """
    return eig(A), (A, eig(A))

def eig_bwd(res, g):
    """
    Parameters
    ----------
    res: residuals from fwd func (tuple)
    g  : vector to backprop (scalar, for some nodes)

    Return
    ------
    vjp: vector-Jacobian product of g with Jacobian (tuple)

    VJP taken from https://github.com/HIPS/autograd/blob/master/autograd/numpy/linalg.py
    """
    A = res[0]
    e, u = res[1] # eigenvalues as 1d array, eigenvectors in columns
    n = e.shape[-1]
    ut = jnp.swapaxes(u, -1, -2)
    
    ge, gu = g
    ge = _matrix_diag(ge)
    
    f = 1/(e[..., jnp.newaxis, :] - e[..., :, jnp.newaxis] + 1.e-20)
    f -= _diag(f)
    
    r1 = f * _dot(ut, gu)
    r2 = -f * (_dot(_dot(ut, jnp.conj(u)), jnp.real(_dot(ut,gu)) * jnp.eye(n)))
    vjp = _dot(_dot(jnp.linalg.inv(ut), ge + r1 + r2), ut)
    
    if not jnp.iscomplexobj(A):
        vjp = jnp.real(vjp)
        # the derivative is still complex for real input (imaginary delta is allowed), real output
        # but the derivative should be real in real input case when imaginary delta is forbidden
    
    return (vjp,)

eig.defvjp(eig_fwd, eig_bwd)
