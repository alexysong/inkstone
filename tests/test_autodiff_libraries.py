import torch # import before numpy to avoid OMP error 15
import numpy as np

import autograd.numpy as npa



# SETTINGS ################################################################################################
np.random.seed(2024)
default_eps = 1e-6
default_abs_tol = 1e-8



# FUNCTIONS ################################################################################################




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