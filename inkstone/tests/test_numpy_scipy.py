"""
Testing that certain numpy and scipy functions with the same name produce the same results

"""

import numpy as np
import scipy as sp



# SETTINGS ################################################################################################
np.random.seed(2024)



# FUNCTIONS ################################################################################################
def test_norm():
    """
    Test that scipy.linalg.norm and numpy.linalg.norm produce the same results
    """
    data = np.ones((100,100)) 
    for axis in [0,1]:
        np_norm = np.linalg.norm(data, axis=axis)
        sp_norm = sp.linalg.norm(data, axis=axis)

        assert (np_norm == sp_norm).all()

def test_ifftshift():
    """
    Test that scipy.fft.ifftshift and numpy.fft.ifftshift produce the same results
    """
    freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
    np_ifftshift = np.fft.ifftshift(np.fft.fftshift(freqs), axes=(0,1))
    sp_ifftshift = sp.fft.ifftshift(sp.fft.fftshift(freqs), axes=(0,1))

    assert (np_ifftshift == sp_ifftshift).all()
    
        