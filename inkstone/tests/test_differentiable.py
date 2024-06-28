import torch # import before numpy to avoid OMP error 15
import numpy as np
import numpy.linalg as nla

import scipy as sp
import scipy.linalg as sla

import autograd.numpy as anp
from autograd import grad as grada, holomorphic_grad as holomorphic_grada

import jax
import jax.numpy as jnp
from jax import grad as gradj

from . import objective_functions
from .finite_diff import finite_diff_grad


# SETTINGS ################################################################################################
np.random.seed(2024)
default_eps = 1e-6
default_rel_tol = 1e-8
default_abs_tol = 1e-8



# AUTOGRAD ################################################################################################
def test_eigen():
    """
    Handle both eigenvector, eigenvalue output from eig()
    """
    data = np.random.uniform(low=-1,high=1,size=(3,3)) + 1j*np.random.uniform(low=-1,high=1,size=(3,3))
    auto_diff_result = holomorphic_grada(objective_functions.eigen)(data)
    finite_diff_result = finite_diff_grad(objective_functions.eigen, data, default_eps)
    abs_error = np.abs(auto_diff_result-finite_diff_result)
    
    assert np.max(abs_error)<default_abs_tol

def test_pi():
    """
    Test if np.pi can be differentiated without anp conversion
    """
    data = np.ones(5)
    auto_diff_result = grada(objective_functions.scaled_sum)(data)
    symbolic_result = np.pi*data
    finite_diff_result = finite_diff_grad(objective_functions.scaled_sum, data, default_eps)
    
    abs_error = np.abs(auto_diff_result-symbolic_result)
    abs_finite_diff_error = np.abs(auto_diff_result-finite_diff_result)
    assert np.max(abs_error)<default_abs_tol and np.max(abs_finite_diff_error)<default_abs_tol 

def test_dot_method_error():
    """
    Test if .method() notation is differentiable by autograd, using .dot()
    """
    x = np.ones(5,dtype=np.float64)
    y = 2*np.ones(5)
    try:
        auto_diff_result = grada(objective_functions.dot_product)(x,y)
        assert False
    except AttributeError:
        assert True

def test_reshape_method():
    """
    Test if .method() notation is differentiable by autograd, using .reshape()
    """
    A = np.array([[1,2,3],[4,5,6]],dtype=np.float64)
    try:
        auto_diff_result = grada(objective_functions.reshape_2x3_to_3x2)(A)
        assert True
    except AttributeError:
        assert False

def test_sla_lu_solve():
    """
    Test if sla.lu_solve is differentiable by autograd
    """
    A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]], dtype=np.float64)
    b = np.array([1, 1, 1, 1], dtype=np.float64)
    
    try:
        auto_diff_result = grada(objective_functions.sum_of_lu_solve)(A,b)
        assert False
    except ValueError:
        assert True



# JAX ################################################################################################
def test_tolist():
    """
    Test if jax can differentiate through .tolist() method
    """
    data = jnp.arange(0,5,1, dtype=jnp.float64)
    try:
        auto_diff_result = gradj(objective_functions.sinxsq)(data)
        assert False
    except jax.errors.ConcretizationTypeError:
        assert True

def test_abs():
    """
    Test the jax return around x=0 when differentiating jnp.abs()
    """
    data = [-0.1, 0., 0.1]
    abs_grad = gradj(jnp.abs)
    auto_diff_result = [abs_grad(x) for x in data]
    answer = [-1, 1, 1]
    assert np.allclose(auto_diff_result, answer, rtol=default_rel_tol, atol=default_abs_tol)
    
def test_jn():
    """
    Test if scipy.special.jn is differentiable by JAX
    """
    x = jnp.linspace(-5,5,100)

    try:
        auto_diff_result = grada(objective_functions.j1)(x)
        assert False
    except TypeError:
        assert True

def test_j0_custom():
    """
    Test j0 custom jax vjp 
    """
    from ..primitives import j0
    data = np.random.uniform(low=-1,high=1,size=(10,))
    def j0_sum(x):
        return jnp.sum(j0(x))
    auto_diff_result = gradj(j0_sum)(data)
    finite_diff_result = finite_diff_grad(j0_sum, data, default_eps)
    assert np.allclose(auto_diff_result, finite_diff_result, rtol=default_rel_tol, atol=default_abs_tol)

def test_j1_custom():
    """
    Test j1 custom jax vjp 
    """
    from ..primitives import j1
    data = np.random.uniform(low=-1,high=1,size=(10,))
    def j1_sum(x):
        return jnp.sum(j1(x))
    auto_diff_result = gradj(j1_sum)(data)
    finite_diff_result = finite_diff_grad(j1_sum, data, default_eps)
    assert np.allclose(auto_diff_result, finite_diff_result, rtol=default_rel_tol, atol=default_abs_tol)



# TORCH ################################################################################################
def test_float_torch():
    """
    Test if np.pi as float can be used with torch functions without conversion
    """
    data = torch.ones(5, requires_grad=True)
    output = objective_functions.scaled_sum_torch(data)
    output.backward()
    auto_diff_result = data.grad
    symbolic_result = np.pi*data
    
    auto_diff_result = auto_diff_result.detach().numpy()
    symbolic_result = symbolic_result.detach().numpy()
    finite_diff_result = finite_diff_grad(objective_functions.scaled_sum, data.detach().numpy(), default_eps)
    
    abs_error = np.abs(auto_diff_result-symbolic_result)
    abs_finite_diff_error = np.abs(auto_diff_result-finite_diff_result)
    assert np.max(abs_error)<default_abs_tol and np.max(abs_finite_diff_error)<1e-6 

def test_np_torch():
    """
    Test if np.pi as np.float64 can be used with torch functions without conversion
    """
    data = torch.ones(5, requires_grad=True)
    output = objective_functions.scaled_sum_torch(data)
    output.backward()
    auto_diff_result = data.grad
    symbolic_result = np.float64(np.pi)*data
    
    auto_diff_result = auto_diff_result.detach().numpy()
    symbolic_result = symbolic_result.detach().numpy()
    
    abs_error = np.abs(auto_diff_result-symbolic_result)
    assert np.max(abs_error)<default_abs_tol