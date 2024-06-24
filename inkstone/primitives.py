import numpy as np
import scipy as sp
import jax.numpy as jnp

from jax import custom_vjp
from autograd.extend import defvjp,primitive



# JAX ################################################################################################################################################################################################
@custom_vjp
def j0(x):
    return sp.special.j0(x)

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
    j0_jac = jnp.diag(sp.special.jvp(0, x, n=1))
    vjp = g*j0_jac
    return (vjp,)

j0.defvjp(j0_fwd, j0_bwd)
