import pytest

import torch # import before numpy to avoid OMP error 15
import numpy as np
import autograd.numpy as anp
import jax.numpy as jnp

from .. import GenericBackend


# SETTINGS ################################################################################################
np.random.seed(2024)



# PYTHON IN-PLACE ################################################################################################
nb = GenericBackend.GenericBackend("numpy")
def test_numpy_backend_selected():
    """
    Ensure the numpy backend is selected
    """
    assert nb.backend == "numpy"

def test_indexAssign_numpy_integer_index():
    """
    Test indexAssign for numpy backend with integer index
    """
    data = nb.arange(10) 
    idx = 0
    replacement = 30
    data = nb.indexAssign(data,idx,replacement)
    
    assert data[idx] == replacement
        
def test_indexAssign_numpy_slice_index():
    """
    Test indexAssign for numpy backend with slice index
    """
    data = nb.arange(10) 
    idx = slice(3,5)
    replacement = 30
    data = nb.indexAssign(data,idx,replacement)
    
    assert (data[idx] == replacement).all()

def test_indexAssign_numpy_slice_all_index():
    """
    Test indexAssign for numpy backend with index replacement for ":"
    """
    data = nb.zeros((10,10)) 
    idx = (slice(None),5)
    replacement = 1
    data = nb.indexAssign(data,idx,replacement)
    
    assert (data[idx] == replacement).all()
        
def test_indexAssign_numpy_truth_index():
    """
    Test indexAssign for numpy backend with truth array index
    """
    data = nb.arange(10) 
    idx = data > 5
    replacement = 5
    data = nb.indexAssign(data,idx,replacement)
    
    assert (data[idx] == replacement).all()
        
def test_inPlaceMultiply_numpy_arrays():
    """
    Test inPlaceMultiply for numpy backend by multiplying two arrays
    """
    array1 = nb.arange(10) 
    array2 = nb.arange(10) 
    array1 = nb.inPlaceMultiply(array1,array2)
    
    assert (array1 == np.power(np.arange(10),2)).all()

def test_assignAndMultiply_numpy_truth_index():
    """
    Test assignAndMultiply for numpy backend with combined array indices
    """
    data = nb.arange(10) 
    idx = (data > 5) * (data <= 8)
    multiply_by = 10
    data = nb.assignAndMultiply(data, idx, multiply_by)
    
    assert (data == np.array([0,1,2,3,4,5,60,70,80,9])).all()
        

# JAX IN-PLACE ################################################################################################
jb = GenericBackend.GenericBackend("jax")
def test_jax_backend_selected():
    """
    Ensure the jax backend is selected
    """
    assert jb.backend == "jax"
        
def test_indexAssign_jax_integer_index():
    """
    Test indexAssign for jax backend with integer index
    """
    data = jb.arange(10) 
    idx = 0
    replacement = 30
    data = jb.indexAssign(data,idx,replacement) # jax arrays are immutable, so have to reassign data variable
    
    assert data[idx] == replacement
        
def test_indexAssign_jax_slice_index():
    """
    Test indexAssign for jax backend with slice index
    """
    data = jb.arange(10) 
    idx = slice(3,5)
    replacement = 30
    data = jb.indexAssign(data,idx,replacement)
    
    assert (data[idx] == replacement).all()

def test_indexAssign_jax_slice_all_index():
    """
    Test indexAssign for jax backend with index replacement for ":"
    """
    data = jb.zeros((10,10)) 
    idx = (slice(None),5)
    replacement = 1
    data = jb.indexAssign(data,idx,replacement)
    
    assert (data[idx] == replacement).all()
        
def test_indexAssign_jax_truth_index():
    """
    Test indexAssign for jax backend with truth array index
    """
    data = jb.arange(10) 
    idx = data > 5
    replacement = 5
    data = jb.indexAssign(data,idx,replacement)
    
    assert (data[idx] == replacement).all()

def test_inPlaceMultiply_jax_arrays():
    """
    Test inPlaceMultiply for numpy backend by multiplying two arrays
    """
    array1 = jb.arange(10) 
    array2 = jb.arange(10) 
    array1 = jb.inPlaceMultiply(array1,array2)
    
    assert (array1 == jnp.power(jnp.arange(10),2)).all()

def test_assignAndMultiply_jax_truth_index():
    """
    Test assignAndMultiply for jax backend with combined array indices
    """
    data = jb.arange(10) 
    idx = (data > 5) * (data <= 8)
    multiply_by = 10
    data = jb.assignAndMultiply(data, idx, multiply_by)
    
    assert (data == jnp.array([0,1,2,3,4,5,60,70,80,9])).all()        