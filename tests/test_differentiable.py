import torch # import before numpy to avoid OMP error 15
import numpy as np

import autograd.numpy as npa
from autograd import grad as grada, holomorphic_grad as holomorphic_grada

import objective_functions

# import jax.numpy as npj
# from jax import grad as gradj


# SETTINGS ################################################################################################
np.random.seed(2024)
default_eps = 1e-6
default_abs_tol = 1e-8



# FUNCTIONS ################################################################################################
def finite_diff(function, input, step, epsilon):
    """
    Calculate the finite difference for a function at a given point, step and step size
    step must be positive or have positive real/imaginary part
    """
    step = np.array(step)
    step_magnitude = step[step!=0]
    return (function(input+step*epsilon/2) - function(input-step*epsilon/2))/(np.abs(step_magnitude)*epsilon)

def finite_diff_grad(function, input, epsilon):
    output = np.array(function(input))
    input = np.array(input)
    input_shape = input.shape
    input_or_output_is_complex = np.any(np.iscomplex(input)) or np.any(np.iscomplex(output))
    
    if input_or_output_is_complex:
        finite_diff_dtype = np.complex128
    else:
        finite_diff_dtype = np.float64
    
    FD_grad = np.zeros_like(input,dtype=finite_diff_dtype)
    if len(input_shape) <= 1:
        for idx in range(input_shape[0]):
            step = np.zeros_like(input,dtype=finite_diff_dtype)
            step[idx] = 1.
            FD_grad[idx] = np.real(finite_diff(function,input,step,epsilon))

            if input_or_output_is_complex:
                step[idx] = 1j
                FD_grad[idx] -= 1j*np.real(finite_diff(function,input,step,epsilon))
    else:
        for row in range(input_shape[0]):
            for col in range(input_shape[1]):
                step = np.zeros_like(input,dtype=finite_diff_dtype)
                step[row,col] = 1.
                
                FD_grad[row,col] = np.real(finite_diff(function,input,step,epsilon))

                if input_or_output_is_complex:
                    step[row,col] = 1j
                    FD_grad[row,col] -= 1j*np.real(finite_diff(function,input,step,epsilon))
    
    return FD_grad



# AUTOGRAD ################################################################################################
def test_eigen():
    """
    Handle both eigenvector, eigenvalue output from eig()
    """
    data = np.random.uniform(low=-1,high=1,size=(3,3))
    auto_diff_result = holomorphic_grada(objective_functions.eigen)(data)
    finite_diff_result = finite_diff_grad(objective_functions.eigen, data, default_eps)
    abs_error = np.abs(auto_diff_result-finite_diff_result)
    
    assert np.max(abs_error)<default_abs_tol

def test_pi():
    """
    Test if np.pi can be differentiated without npa conversion
    """
    data = np.ones(5)
    auto_diff_result = grada(objective_functions.scaled_sum)(data)
    symbolic_result = np.pi*data
    finite_diff_result = finite_diff_grad(objective_functions.scaled_sum, data, default_eps)
    
    abs_error = np.abs(auto_diff_result-symbolic_result)
    abs_finite_diff_error = np.abs(auto_diff_result-finite_diff_result)
    pass_condition = np.max(abs_error)<default_abs_tol and np.max(abs_finite_diff_error)<default_abs_tol 
    
    assert pass_condition

def test_dot_method_error():
    """
    Test if .method() notation is differentiable by autograd, using .dot()
    """
    x = np.ones(5,dtype=np.float64)
    y = 2*np.ones(5)
    H = np.array([[1,2,3],[4,5,6]])
    try:
        auto_diff_result = grada(objective_functions.dot_product)(x,y)
        pass_condition = False
    except AttributeError:
        pass_condition = True
    
    assert pass_condition

def test_reshape_method():
    """
    Test if .method() notation is differentiable by autograd, using .reshape()
    """
    H = np.array([[1,2,3],[4,5,6]],dtype=np.float64)
    try:
        auto_diff_result = grada(objective_functions.reshape_2x3_to_3x2)(H)
        pass_condition = True
    except AttributeError:
        pass_condition = False
    
    assert pass_condition



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
    pass_condition = np.max(abs_error)<default_abs_tol and np.max(abs_finite_diff_error)<1e-6 
    
    assert pass_condition

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
    pass_condition = np.max(abs_error)<default_abs_tol
    
    assert pass_condition