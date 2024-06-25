import numpy as np
import jax.numpy as jnp
from jax import grad as jax_grad

import pytest
# from ... import GenericBackend
# from ...simulator import Inkstone

from .inkstone_objective_functions import reflection_1d



# SETTINGS ################################################################################################
np.random.seed(2024)
abs_tol = 0
rel_tol = 1e-6
eps = 1e-7


# SETUP ################################################################################################
@pytest.fixture
def params_1d():
    d = 1.
    w = 0.4
    f = 0.2
    permittivity = 12.
    return (d,w,f,permittivity)



# JAX ################################################################################################
## 1D SIMULATION ##
# reflection_1d #
reflection_1d_grad0 = jax_grad(reflection_1d,0)
reflection_1d_grad1 = jax_grad(reflection_1d,1)
reflection_1d_grad2 = jax_grad(reflection_1d,2)
reflection_1d_grad3 = jax_grad(reflection_1d,3)

def test_reflection_1d_grad0(params_1d):
    """
    Test jax gradient of reflection_1d wrt grating thickness matches finite difference
    """
    grating_thicknesses = jnp.linspace(0.1,100,5, dtype=jnp.float64)

    for d in grating_thicknesses:
        params = params_1d[1:]
        FD_grad = (reflection_1d(d+eps/2,*params,num_g=30,backend="jax") - reflection_1d(d-eps/2,*params,num_g=30,backend="jax"))/eps
        AD_grad = reflection_1d_grad0(d,*params,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)
    
def test_reflection_1d_grad1(params_1d):
    """
    Test jax gradient of reflection_1d wrt grating resonator width matches finite difference
    """
    grating_widths = jnp.linspace(0.01,0.99,5, dtype=jnp.float64)

    for w in grating_widths:
        d = params_1d[0]
        params = params_1d[2:]
        FD_grad = (reflection_1d(d,w+eps/2,*params,num_g=30,backend="jax") - reflection_1d(d,w-eps/2,*params,num_g=30,backend="jax"))/eps
        AD_grad = reflection_1d_grad1(d,w,*params,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_reflection_1d_grad2(params_1d):
    """
    Test jax gradient of reflection_1d wrt frequency matches finite difference
    """
    frequencies = jnp.linspace(0.01,0.99,5, dtype=jnp.float64)

    for f in frequencies:
        d,w = params_1d[:2]
        permittivity = params_1d[3]
        FD_grad = (reflection_1d(d,w,f+eps/2,permittivity,num_g=30,backend="jax") - reflection_1d(d,w,f-eps/2,permittivity,num_g=30,backend="jax"))/eps
        AD_grad = reflection_1d_grad2(d,w,f,permittivity,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_reflection_1d_grad3(params_1d):
    """
    Test jax gradient of reflection_1d wrt grating permittivity matches finite difference
    """
    permittivities = jnp.logspace(-2,2,5, dtype=jnp.float64)

    for p in permittivities:
        params = params_1d[:3]
        permittivity = params_1d[3]
        FD_grad = (reflection_1d(*params,permittivity+eps/2,num_g=30,backend="jax") - reflection_1d(*params,permittivity-eps/2,num_g=30,backend="jax"))/eps
        AD_grad = reflection_1d_grad3(*params,permittivity,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)