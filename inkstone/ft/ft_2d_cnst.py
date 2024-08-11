# -*- coding: utf-8 -*-

import inkstone.backends.BackendLoader as bl


def ft_2d_cnst(ks,gb=bl.backend()):
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
    #ksa = gb.parseData(ks)  # nx2 shape
    ks_nm = gb.la.norm(ks, axis=-1)  # 1d array of n. The norm of each k vector
    idx_0 = gb.where(ks_nm == 0)[0]  # index to where k is (0, 0)
    s = 1j * gb.zeros(gb.getSize(ks_nm))
    s = gb.indexAssign(s, idx_0, 1.)
    return gb.castType(s, gb.complex128)
