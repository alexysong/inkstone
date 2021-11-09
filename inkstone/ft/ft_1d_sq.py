# -*- coding: utf-8 -*-

import numpy as np


def ft_1d_sq(width, ks, center=0.):

    """
    Calculate the Fourier transform of a 1d square function.

    The square function looks like this:

           __________
    ______|    1     |____0____

    Parameters
    ----------
    center      :   float
                    center of the square
    width       :   float
                    width of the square
    ks          :   list
                    list of k points

    Returns
    -------
    s           :   list[complex]
                    1D array, Fourier series coefficients at the corresponding points
    """

    ksa = np.array(ks)

    s = np.exp(-1j * center * ksa) * width * np.sinc(ksa * width / 2. / np.pi)
    # note numpy sinc(x) definition is sin(pi x) / (pi x)

    return s.tolist()

