"""
Testing backend custom functions

"""

import pytest

import torch # import before numpy to avoid OMP error 15
import numpy as np
import autograd.numpy as anp
import jax.numpy as jnp

from .. import GenericBackend


# SETTINGS ################################################################################################
np.random.seed(2024)



# NUMPY ################################################################################################
nb = GenericBackend.GenericBackend("numpy")
def test_numpy_backend_selected():
    """
    Ensure the numpy backend is selected
    """
    assert nb.backend == "numpy"

def test_data_numpy_list_of_arrays():
    """
    Test data for numpy backend with list of arrays
    """
    data = [[np.array(1.),np.array(2.)],[np.array(3.),np.array(4.)]] 
    converted_data = nb.data(data)
    
    assert (converted_data == np.array([[1.,2.],[3.,4.]])).all()

        

# JAX ################################################################################################
jb = GenericBackend.GenericBackend("jax")
def test_jax_backend_selected():
    """
    Ensure the jax backend is selected
    """
    assert jb.backend == "jax"

def test_data_jax_list_of_arrays():
    """
    Test data for jax backend with list of arrays
    """
    data = [[jnp.array(1.),jnp.array(2.)],[jnp.array(3.),jnp.array(4.)]] 
    converted_data = jb.data(data)
    
    assert (converted_data == jnp.array([[1.,2.],[3.,4.]])).all()