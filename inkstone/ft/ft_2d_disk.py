# -*- coding: utf-8 -*-

from GenericBackend import genericBackend as gb
from scipy.special import jn



def ft_2d_disk(r, ks, center=(0,0),gb=gb): 
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
    if not ks:
        return []
    
    ksa = gb.parseData(ks,dtype=gb.complex128)  # nx2 shape
    ks_nm = gb.la.norm(ksa, axis=-1)  # 1d array of n. The norm of each k vector
    idx_0 = gb.where(ks_nm == 0)[0]  # index to where k is (0, 0)
    idx_i = gb.where(ks_nm != 0)[0]  # index to where k is not (0, 0)
    ks_nm1 = gb.delete(ks_nm, idx_0)

    ksa1 = gb.delete(ksa, idx_0, axis=0)  # new ks array that doesn't contain (0, 0)

    cent = gb.parseData(center,dtype=gb.complex128)

    s = 1j * gb.zeros(gb.getSize(ks_nm),dtype=gb.complex128)
 
    s[idx_i] = 2 * gb.pi * r * jn(1, r * ks_nm1) / ks_nm1 * gb.exp(-1j * ksa1 @ cent)
    s[idx_0] = gb.pi * r**2

    return s.tolist()

