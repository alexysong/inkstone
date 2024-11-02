# -*- coding: utf-8 -*-

from inkstone.backends.BackendRegistry import backend


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
    #ksa = gb.data(ks)  # nx2 shape
    gb = backend()
    ks_nm = gb.norm(ks, dim=-1)  # 1d array of n. The norm of each k vector
    idx_0 = gb.where(ks_nm == 0)[0]  # index to where k is (0, 0)
    s = 1j * gb.zeros(gb.getSize(ks_nm))
    s = gb.indexAssign(s, idx_0, 1.)
    return gb.castType(s, gb.complex128)
