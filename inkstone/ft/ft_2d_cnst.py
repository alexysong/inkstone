# -*- coding: utf-8 -*-

import numpy as np


def ft_2d_cnst(ks):
    """
    calculate the fourier transform of a constant 1. Results in delta function.

    Parameters
    ----------
    ks      :   list[tuple[float, float]]
                list of (kx, ky) points
    Returns
    -------
    s       :   list[complex]
                Fourier coefficient at the input ks positions
    """
    ksa = np.array(ks)  # nx2 shape
    ks_nm = np.linalg.norm(ksa, axis=-1)  # 1d array of n. The norm of each k vector
    idx_0 = np.where(ks_nm == 0)[0]  # index to where k is (0, 0)

    s = 1j * np.zeros(ks_nm.size)
    s[idx_0] = 1.

    return s.tolist()
