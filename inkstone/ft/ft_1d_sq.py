# -*- coding: utf-8 -*-
from GenericBackend import genericBackend as gb


def ft_1d_sq(width, ks, center=0. , gb=gb):

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
    if width <= 0:
       raise ValueError("No zero or negative width")

    ksa = gb.parseData(ks)

    s = gb.exp(-1j * center * ksa) * width * gb.sinc(ksa * width / 2. / gb.pi)
    # note numpy sinc(x) definition is sin(pi x) / (pi x)

    return s.tolist()

