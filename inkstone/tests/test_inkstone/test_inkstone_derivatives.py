"""
Testing automatic differentiation of inkstone-based objective functions matches finite difference derivatives.

"""

import numpy as np
import jax.numpy as jnp
from jax import grad as jax_grad

import pytest
# from ... import GenericBackend
# from ...simulator import Inkstone

from .inkstone_objective_functions import reflection_1layer_1d, abs_Ey_1layer_1d, reflection_2layer_1d, abs_Ey_2layer_1d
from .inkstone_objective_functions import reflection_1layer_2d, abs_Ey_1layer_2d


# SETTINGS ################################################################################################
np.random.seed(2024)
abs_tol = 1e-5
rel_tol = 1e-5
eps = 1e-6


# SETUP ################################################################################################
## PARAMETERS ##
@pytest.fixture
def params_1layer_1d():
    d = 1.
    w = 0.4
    f = 0.2
    p = 12.
    z = d/2
    return [d,w,f,p,z]

@pytest.fixture
def params_2layer_1d():
    d1 = 0.5
    d2 = 10.
    w1 = 0.4
    w2 = 0.9
    p1 = 12.
    p2 = 0.05
    f = 10.
    theta = 25. # degrees
    z = 1.05*d1
    return [d1,d2,w1,w2,p1,p2,f,theta,z]

@pytest.fixture
def params_1layer_2d():
    d = 1.
    r = 0.4
    f = 0.2
    p = 12.
    theta = 0.
    phi = 0.
    x = r/2
    z = d/2
    return [d,r,f,p,theta,phi,x,z]


## GRADIENT FUNCTIONS ##
@pytest.fixture
def reflection_1layer_1d_grad():
    def _reflection_1layer_1d_grad(derivative_argnum=0):
        return jax_grad(reflection_1layer_1d, derivative_argnum)

    return _reflection_1layer_1d_grad

@pytest.fixture
def abs_Ey_1layer_1d_grad():
    def _abs_Ey_1layer_1d_grad(derivative_argnum=0):
        return jax_grad(abs_Ey_1layer_1d, derivative_argnum)

    return _abs_Ey_1layer_1d_grad

@pytest.fixture
def reflection_2layer_1d_grad():
    def _reflection_2layer_1d_grad(derivative_argnum=0):
        return jax_grad(reflection_2layer_1d, derivative_argnum)

    return _reflection_2layer_1d_grad

@pytest.fixture
def abs_Ey_2layer_1d_grad():
    def _abs_Ey_2layer_1d_grad(derivative_argnum=0):
        return jax_grad(abs_Ey_2layer_1d, derivative_argnum)

    return _abs_Ey_2layer_1d_grad


@pytest.fixture
def reflection_1layer_2d_grad():
    def _reflection_1layer_2d_grad(derivative_argnum=0):
        return jax_grad(reflection_1layer_2d, derivative_argnum)

    return _reflection_1layer_2d_grad

@pytest.fixture
def abs_Ey_1layer_2d_grad():
    def _abs_Ey_1layer_2d_grad(derivative_argnum=0):
        return jax_grad(abs_Ey_1layer_2d, derivative_argnum)

    return _abs_Ey_1layer_2d_grad



# JAX ################################################################################################
## 1D SIMULATION ##
# reflection_1layer_1d #
def test_reflection_1layer_1d_grad_d(params_1layer_1d, reflection_1layer_1d_grad):
    """
    Test jax gradient of reflection_1layer_1d wrt grating thickness matches finite difference
    """
    autodiffgrad = reflection_1layer_1d_grad(derivative_argnum=0)
    grating_thicknesses = jnp.linspace(0.1,100,5, dtype=jnp.float64)
    params = params_1layer_1d[1:4]
    for d in grating_thicknesses:
        FD_grad = (reflection_1layer_1d(d+eps/2,*params,num_g=30,backend="jax") - reflection_1layer_1d(d-eps/2,*params,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(d,*params,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)
    
def test_reflection_1layer_1d_grad_w(params_1layer_1d, reflection_1layer_1d_grad):
    """
    Test jax gradient of reflection_1layer_1d wrt grating resonator width matches finite difference
    """
    autodiffgrad = reflection_1layer_1d_grad(derivative_argnum=1)
    grating_widths = jnp.linspace(0.01,0.99,5, dtype=jnp.float64)
    d = params_1layer_1d[0]
    params = params_1layer_1d[2:4]
    for w in grating_widths:
        FD_grad = (reflection_1layer_1d(d,w+eps/2,*params,num_g=30,backend="jax") - reflection_1layer_1d(d,w-eps/2,*params,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(d,w,*params,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_reflection_1layer_1d_grad_f(params_1layer_1d, reflection_1layer_1d_grad):
    """
    Test jax gradient of reflection_1layer_1d wrt frequency matches finite difference
    """
    autodiffgrad = reflection_1layer_1d_grad(derivative_argnum=2)
    frequencies = jnp.linspace(0.01,0.99,5, dtype=jnp.float64)
    d,w = params_1layer_1d[:2]
    p = params_1layer_1d[3]
    for f in frequencies:
        FD_grad = (reflection_1layer_1d(d,w,f+eps/2,p,num_g=30,backend="jax") - reflection_1layer_1d(d,w,f-eps/2,p,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(d,w,f,p,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_reflection_1layer_1d_grad_p(params_1layer_1d, reflection_1layer_1d_grad):
    """
    Test jax gradient of reflection_1layer_1d wrt grating permittivity matches finite difference
    """
    autodiffgrad = reflection_1layer_1d_grad(derivative_argnum=3)
    permittivities = jnp.logspace(-2,2,5, dtype=jnp.float64)
    params = params_1layer_1d[:3]
    for p in permittivities:
        FD_grad = (reflection_1layer_1d(*params,p+eps/2,num_g=30,backend="jax") - reflection_1layer_1d(*params,p-eps/2,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(*params,p,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)


# abs_Ey_1layer_1d #
def test_abs_Ey_1layer_1d_grad_d(params_1layer_1d, abs_Ey_1layer_1d_grad):
    """
    Test jax gradient of abs_Ey_1layer_1d wrt grating thickness matches finite difference
    """
    autodiffgrad = abs_Ey_1layer_1d_grad(derivative_argnum=0)
    grating_thicknesses = jnp.linspace(0.1,100,5, dtype=jnp.float64)
    params = params_1layer_1d[1:]
    for d in grating_thicknesses:
        FD_grad = (abs_Ey_1layer_1d(d+eps/2,*params,num_g=30,backend="jax") - abs_Ey_1layer_1d(d-eps/2,*params,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(d,*params,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)
    
def test_abs_Ey_1layer_1d_grad_w(params_1layer_1d, abs_Ey_1layer_1d_grad):
    """
    Test jax gradient of abs_Ey_1layer_1d wrt grating resonator width matches finite difference
    """
    autodiffgrad = abs_Ey_1layer_1d_grad(derivative_argnum=1)
    grating_widths = jnp.linspace(0.01,0.99,5, dtype=jnp.float64)
    d = params_1layer_1d[0]
    params = params_1layer_1d[2:]
    for w in grating_widths:
        FD_grad = (abs_Ey_1layer_1d(d,w+eps/2,*params,num_g=30,backend="jax") - abs_Ey_1layer_1d(d,w-eps/2,*params,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(d,w,*params,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_abs_Ey_1layer_1d_grad_f(params_1layer_1d, abs_Ey_1layer_1d_grad):
    """
    Test jax gradient of abs_Ey_1layer_1d wrt frequency matches finite difference
    """
    autodiffgrad = abs_Ey_1layer_1d_grad(derivative_argnum=2)
    frequencies = jnp.linspace(0.01,0.99,5, dtype=jnp.float64)
    d,w = params_1layer_1d[:2]
    p, z = params_1layer_1d[3:5]
    for f in frequencies:
        FD_grad = (abs_Ey_1layer_1d(d,w,f+eps/2,p,z,num_g=30,backend="jax") - abs_Ey_1layer_1d(d,w,f-eps/2,p,z,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(d,w,f,p,z,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_abs_Ey_1layer_1d_grad_p(params_1layer_1d, abs_Ey_1layer_1d_grad):
    """
    Test jax gradient of abs_Ey_1layer_1d wrt grating permittivity matches finite difference
    """
    autodiffgrad = abs_Ey_1layer_1d_grad(derivative_argnum=3)
    permittivities = jnp.logspace(-2,2,5, dtype=jnp.float64)
    params = params_1layer_1d[:3]
    z = params_1layer_1d[4]
    for p in permittivities:
        FD_grad = (abs_Ey_1layer_1d(*params,p+eps/2,z,num_g=30,backend="jax") - abs_Ey_1layer_1d(*params,p-eps/2,z,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(*params,p,z,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_abs_Ey_1layer_1d_grad_z(params_1layer_1d, abs_Ey_1layer_1d_grad):
    """
    Test jax gradient of abs_Ey_1layer_1d wrt z position matches finite difference
    """
    autodiffgrad = abs_Ey_1layer_1d_grad(derivative_argnum=4)
    params = params_1layer_1d[:4]
    z_positions = jnp.linspace(-1,1.05*params[0],5, dtype=jnp.float64)
    for z in z_positions:
        FD_grad = (abs_Ey_1layer_1d(*params,z+eps/2,num_g=30,backend="jax") - abs_Ey_1layer_1d(*params,z-eps/2,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(*params,z,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)


# reflection_2layer_1d #
def test_reflection_2layer_1d_grad_d1(params_2layer_1d, reflection_2layer_1d_grad):
    """
    Test jax gradient of reflection_2layer_1d wrt first layer grating thickness matches finite difference
    """
    autodiffgrad = reflection_2layer_1d_grad(derivative_argnum=0)
    grating_thicknesses = jnp.linspace(0.1,100,5, dtype=jnp.float64)
    params = params_2layer_1d[1:-1] # excluding z parameter at params_2layer_1d[-1]
    for d1 in grating_thicknesses:
        FD_grad = (reflection_2layer_1d(d1+eps/2,*params,order=-1,num_g=30,backend="jax") 
                   - reflection_2layer_1d(d1-eps/2,*params,order=-1,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(d1,*params,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=10*rel_tol, atol=abs_tol)

def test_reflection_2layer_1d_grad_w2(params_2layer_1d, reflection_2layer_1d_grad):
    """
    Test jax gradient of reflection_2layer_1d wrt second layer grating width matches finite difference
    """
    autodiffgrad = reflection_2layer_1d_grad(derivative_argnum=3)
    grating_widths = jnp.linspace(0.01,0.99,5, dtype=jnp.float64)
    params_left = params_2layer_1d[:3] 
    params_right = params_2layer_1d[4:-1] 
    for w2 in grating_widths:
        FD_grad = (reflection_2layer_1d(*params_left,w2+eps/2,*params_right,order=-1,num_g=30,backend="jax") 
                   - reflection_2layer_1d(*params_left,w2-eps/2,*params_right,order=-1,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,w2,*params_right,order=-1,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=1000*rel_tol, atol=abs_tol)

def test_reflection_2layer_1d_grad_p2(params_2layer_1d, reflection_2layer_1d_grad):
    """
    Test jax gradient of reflection_2layer_1d wrt second layer permittivity matches finite difference
    """
    autodiffgrad = reflection_2layer_1d_grad(derivative_argnum=5)
    permittivities = jnp.logspace(-2,2,5, dtype=jnp.float64)
    params_left = params_2layer_1d[:5] 
    params_right = params_2layer_1d[6:-1] 
    for p2 in permittivities:
        FD_grad = (reflection_2layer_1d(*params_left,p2+eps/2,*params_right,order=-1,num_g=30,backend="jax") 
                   - reflection_2layer_1d(*params_left,p2-eps/2,*params_right,order=-1,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,p2,*params_right,order=-1,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_reflection_2layer_1d_grad_theta(params_2layer_1d, reflection_2layer_1d_grad):
    """
    Test jax gradient of reflection_2layer_1d wrt incident angle matches finite difference

    AssertionError: 
        AD_grad = nan, FD_grad = 0.1961769746719466
    JL's progress: 
        The error seems to come from JAX being unable to take the jnp.sqrt of a TracedConcreteArray(0.) in params.py
        Specifically, where the argument of the square root can be zero:
            _calc_q0(): q0 = self.gb.sqrt(q02)
            _calc_angles(): sthe = self.gb.sqrt(1 - cthe**2 + 0j) 
        JAX has no issue with jnp.sqrt(0.) = 0 or jnp.sqrt(0.+0j) = 0, but the square root of the Traced array creates NaN
        Attempted fix: 
            Converting NaNs to zero, e.g. sthe = self.gb.indexAssign(sthe, self.gb.isnan(sthe), 0)
            This does not solve the issue, since the final gradient is still NaN (it is possible that JAX cannot differentiate gb.isnan)
        Alternative attempted fix: 
            I also tried converting all instances of zero in q02 and 1-cthe**2+0j to a small positive number 
            1.e-15+0j before taking the square root. Gives a non-NaN gradient, but final AD_grad is inconsistent with FD_grad.
            I am not sure if this is allowed in the physics, or computationally valid since the sqrt of a small number becomes larger
        A better fix would not allow modifying values.
    """
    autodiffgrad = reflection_2layer_1d_grad(derivative_argnum=7)
    angles = jnp.linspace(-30.,30.,5, dtype=jnp.float64)
    params_left = params_2layer_1d[:7] 
    eps = 1e-6 
    for th in angles:
        FD_grad = (reflection_2layer_1d(*params_left,th+eps/2,order=-1,num_g=30,backend="jax") 
                   - reflection_2layer_1d(*params_left,th-eps/2,order=-1,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,th,order=-1,num_g=30,backend="jax")
        # print(AD_grad,FD_grad) 
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)


# abs_Ey_2layer_1d
def test_abs_Ey_2layer_1d_grad_d2(params_2layer_1d, abs_Ey_2layer_1d_grad):
    """
    Test jax gradient of abs_Ey_2layer_1d wrt second layer grating thickness matches finite difference

    AssertionError:
        Test case fails likely because |Ey| does not seem to be continuous with respect to grating thickness.
        In the 1D, 1 layer case, the fields were periodic in d because of the exp(iq(d-z)) Fourier field dependence
        Why is |Ey| no longer continuous (let alone periodic) in d2 in the 1D, 2 layer case?
            Can check discontinuity by plotting |Ey| vs d2 for a small d2 range and increasing the number of plot points.
    """
    autodiffgrad = abs_Ey_2layer_1d_grad(derivative_argnum=1)
    # grating_thicknesses = jnp.linspace(0.1,100,5, dtype=jnp.float64) # FD diverges at d2 = 0.1
    grating_thicknesses = jnp.linspace(1,100,5, dtype=jnp.float64)
    params_left = params_2layer_1d[:1] 
    params_right = params_2layer_1d[2:]
    eps = 1e-6
    for d2 in grating_thicknesses:
        FD_grad = (abs_Ey_2layer_1d(*params_left,d2+eps/2,*params_right,num_g=30,backend="jax") 
                   - abs_Ey_2layer_1d(*params_left,d2-eps/2,*params_right,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,d2,*params_right,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)
    
def test_abs_Ey_2layer_1d_grad_z(params_2layer_1d, abs_Ey_2layer_1d_grad):
    """
    Test jax gradient of abs_Ey_2layer_1d wrt second layer grating thickness matches finite difference
    """
    autodiffgrad = abs_Ey_2layer_1d_grad(derivative_argnum=8)
    params_left = params_2layer_1d[:8]
    d1 = params_left[0]
    d2 = params_left[1]
    # z_positions = [-1., d1/2, d1, d1+d2/2, d1+d2, 1.05*(d1+d2)] # FD diverges at d1, d1+d2
    z_positions = [-1., d1/2, d1+d2/2, 1.05*(d1+d2)]
    for z in z_positions:
        FD_grad = (abs_Ey_2layer_1d(*params_left,z+eps/2,num_g=30,backend="jax") 
                   - abs_Ey_2layer_1d(*params_left,z-eps/2,num_g=30,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,z,num_g=30,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=10*rel_tol, atol=abs_tol)



## 2D SIMULATION ##
# reflection_1layer_2d #
def test_reflection_1layer_2d_grad_d(params_1layer_2d, reflection_1layer_2d_grad):
    """
    Test jax gradient of reflection_1layer_2d wrt grating thickness matches finite difference
    """
    autodiffgrad = reflection_1layer_2d_grad(derivative_argnum=0)
    grating_thicknesses = jnp.linspace(0.1,100,5, dtype=jnp.float64)
    params = params_1layer_2d[1:-2] # excluding x, z parameters
    for d in grating_thicknesses:
        FD_grad = (reflection_1layer_2d(d+eps/2,*params,num_g=100,backend="jax") 
                   - reflection_1layer_2d(d-eps/2,*params,num_g=100,backend="jax"))/eps
        AD_grad = autodiffgrad(d,*params,num_g=100,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_reflection_1layer_2d_grad_r(params_1layer_2d, reflection_1layer_2d_grad):
    """
    Test jax gradient of reflection_1layer_2d wrt hole radius matches finite difference
    """
    autodiffgrad = reflection_1layer_2d_grad(derivative_argnum=1)
    hole_radii = jnp.linspace(0.01,0.99,5, dtype=jnp.float64)
    params_left = params_1layer_2d[:1] 
    params_right = params_1layer_2d[2:-2] 
    for r in hole_radii:
        FD_grad = (reflection_1layer_2d(*params_left,r+eps/2,*params_right,num_g=100,backend="jax") 
                   - reflection_1layer_2d(*params_left,r-eps/2,*params_right,num_g=100,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,r,*params_right,num_g=100,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_reflection_1layer_2d_grad_theta(params_1layer_2d, reflection_1layer_2d_grad):
    """
    Test jax gradient of reflection_1layer_2d wrt incident theta matches finite difference
    """
    autodiffgrad = reflection_1layer_2d_grad(derivative_argnum=4)
    angles = jnp.linspace(-30.,30.,5, dtype=jnp.float64)
    params_left = params_1layer_2d[:4] 
    params_right = params_1layer_2d[5:-2] 
    for th in angles:
        FD_grad = (reflection_1layer_2d(*params_left,th+eps/2,*params_right,num_g=100,backend="jax") 
                   - reflection_1layer_2d(*params_left,th-eps/2,*params_right,num_g=100,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,th,*params_right,num_g=100,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def test_reflection_1layer_2d_grad_phi(params_1layer_2d, reflection_1layer_2d_grad):
    """
    Test jax gradient of reflection_1layer_2d wrt incident phi matches finite difference
    """
    autodiffgrad = reflection_1layer_2d_grad(derivative_argnum=5)
    angles = jnp.linspace(-30.,30.,5, dtype=jnp.float64)
    params_left = params_1layer_2d[:5] 
    for ph in angles:
        FD_grad = (reflection_1layer_2d(*params_left,ph+eps/2,num_g=100,backend="jax") 
                   - reflection_1layer_2d(*params_left,ph-eps/2,num_g=100,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,ph,num_g=100,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)


# abs_Ey_1layer_2d
def test_abs_Ey_1layer_2d_grad_x(params_1layer_2d, abs_Ey_1layer_2d_grad):
    """
    Test jax gradient of abs_Ey_1layer_2d wrt x position matches finite difference
    """
    autodiffgrad = abs_Ey_1layer_2d_grad(derivative_argnum=6)
    x_positions = jnp.linspace(0.01,1.,5, dtype=jnp.float64)
    params_left = params_1layer_2d[:6] 
    params_right = params_1layer_2d[7:] 
    for x in x_positions:
        FD_grad = (abs_Ey_1layer_2d(*params_left,x+eps/2,*params_right,num_g=100,backend="jax") 
                   - abs_Ey_1layer_2d(*params_left,x-eps/2,*params_right,num_g=100,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,x,*params_right,num_g=100,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)
    
def test_abs_Ey_1layer_2d_grad_z(params_1layer_2d, abs_Ey_1layer_2d_grad):
    """
    Test jax gradient of abs_Ey_1layer_2d wrt z position matches finite difference
    """
    autodiffgrad = abs_Ey_1layer_2d_grad(derivative_argnum=7)
    params_left = params_1layer_2d[:7] 
    d = params_left[0]
    z_positions = [-1., 0., d/2, d, 1.05*d, 10*d]
    for z in z_positions:
        FD_grad = (abs_Ey_1layer_2d(*params_left,z+eps/2,num_g=100,backend="jax") 
                   - abs_Ey_1layer_2d(*params_left,z-eps/2,num_g=100,backend="jax"))/eps
        AD_grad = autodiffgrad(*params_left,z,num_g=100,backend="jax")
        assert np.allclose(AD_grad, FD_grad, rtol=10000*rel_tol, atol=abs_tol)