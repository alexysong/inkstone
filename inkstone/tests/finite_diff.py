import numpy as np

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
            FD_grad[idx] = np.real(finite_diff(function,input,step,epsilon)).item()

            if input_or_output_is_complex:
                step[idx] = 1j
                FD_grad[idx] -= 1j*np.real(finite_diff(function,input,step,epsilon)).item()
    else:
        for row in range(input_shape[0]):
            for col in range(input_shape[1]):
                step = np.zeros_like(input,dtype=finite_diff_dtype)
                step[row,col] = 1.
                FD_grad[row,col] = np.real(finite_diff(function,input,step,epsilon)).item()

                if input_or_output_is_complex:
                    step[row,col] = 1j
                    FD_grad[row,col] -= 1j*np.real(finite_diff(function,input,step,epsilon)).item()
    
    return FD_grad