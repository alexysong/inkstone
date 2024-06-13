# -*- coding: utf-8 -*-

from GenericBackend import genericBackend as gb


def ft_2d_cnst(ks,gb=gb):
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
    ksa = gb.parseData(ks)  # nx2 shape
    ks_nm = gb.la.norm(ksa, axis=-1)  # 1d array of n. The norm of each k vector
    idx_0 = gb.where(ks_nm == 0)[0]  # index to where k is (0, 0)
    s = 1j * gb.zeros(gb.getSize(ks_nm))
    s[idx_0] = 1.

    return s.tolist()
