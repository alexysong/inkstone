# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import jn
from typing import Union


def ft_2d_disk(r, ks, center=None):
    """
    calculate the fourier transform of a function, its value is 1 inside a disk, outside its value is 0.

    Parameters
    ----------
    r               :   float
                        vertices
    ks              :   list[tuple[float, float]]
                        list of (kx, ky) points. The
    center          :   tuple[float, float]
                        the center of the disk

    Returns
    -------
    s       :   list[complex]
                1d array, Fourier coefficient at the input ks positions
    """
    ksa = np.array(ks)  # nx2 shape
    ks_nm = np.linalg.norm(ksa, axis=-1)  # 1d array of n. The norm of each k vector
    idx_0 = np.where(ks_nm == 0)[0]  # index to where k is (0, 0)
    idx_i = np.where(ks_nm != 0)[0]  # index to where k is not (0, 0)
    ks_nm1 = np.delete(ks_nm, idx_0)

    ksa1 = np.delete(ksa, idx_0, axis=0)  # new ks array that doesn't contain (0, 0)

    cent = np.array(center)

    s = 1j * np.zeros(ks_nm.size)
    s[idx_i] = 2 * np.pi * r * jn(1, r * ks_nm1) / ks_nm1 * np.exp(-1j * ksa1 @ cent)
    s[idx_0] = np.pi * r**2

    return s.tolist()

