import torch # import before numpy to avoid OMP error 15
import numpy as np
import autograd.numpy as anp
import jax.numpy as jnp



# SETTINGS ################################################################################################
np.random.seed(2024)



# FUNCTIONS ################################################################################################
def test_jax_index_assignment():
    """
    Ensure index assignment is disallowed with jax
    """
    data = jnp.arange(10)
    
    try:
        data[3] = 30
        assert False
    except TypeError:
        assert True

def test_jax_at():
    """
    Test jax's index assignment alternative data.at[idx].set(x) works
    """
    data = jnp.arange(10) 
    data = data.at[3].set(30)
    
    assert data[3] == 30
        
def test_jax_at_slicing():
    """
    Test jax's index assignment alternative data.at[idx].set(x) with slice index works
    """
    data = jnp.arange(10) 
    idx = slice(1,3)
    data = data.at[idx].set(30)
    
    assert (data[idx] == 30).all()
        
def test_jax_at_multidim():
    """
    Test jax's index assignment alternative data.at[idx].set(x) with multidimensional indexing works
    """
    data = jnp.ones((3,5,10)) 
    idx = jnp.array([2,4,9])
    # idx = (2,4,9) # tuple also works 
    data = data.at[idx].set(30)
    
    assert (data[idx] == 30).all()

def test_jax_immutable():
    """
    Test that jax's arrays are immutable
    """
    data = jnp.ones((3,5,10)) 
    data_copy = data
    data_copy = jnp.sum(data_copy)

    assert (data == jnp.ones((3,5,10))).all()



# METHODS WITH THE SAME NAME ################################################################################################
def test_any_check():
    """
    Determine which .any() method is used from torch or numpy
    """
    np_data = np.array([1,2])
    torch_data = torch.tensor([1,2])
    
    assert np_data.any() and torch_data.any()

def test_any_empty_check():
    """
    Determine if .any() correctly identifies empty arrays/tensors
    """
    np_data = np.array([])
    torch_data = torch.tensor([])
    
    assert (not np_data.any()) and (not torch_data.any())