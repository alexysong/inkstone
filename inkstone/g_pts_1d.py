# -*- coding: utf-8 -*-

from GenericBackend import genericBackend as gb
# from numpy import linalg as la


def g_pts_1d(num_g, b,gb=gb):
    """
    given number of lattice points and one 2D lattice vector, return the list of lattice points and the index of lattice points.

    Parameters
    ----------
    num_g       :   int
                    target total number of lattice points
    b           :   tuple[float, float]
                    lattice vector

    Returns
    -------
    k_pts       :   list[tuple[float, float]]
                    the k points
    idx         :   list[int]
                    the indices of the k points, i.e. k = idx * b

    """

    imax = int(num_g / 2)

    idx = [i - 2 * i * j for i in range(imax+1) for j in range(2)][1:]
    # This gives [0, 1, -1, 2, -2, ..., imax, -imax]

    k_pts = [(b[0] * i, b[1] * i) for i in idx]

    return k_pts, idx

