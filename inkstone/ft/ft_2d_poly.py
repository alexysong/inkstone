# -*- coding: utf-8 -*-

"""
Calculate the Fourier transform of 2d polygon.

Reference:
K. McInturff and P.S. Simon, "The Fourier transform of linearly varying functions with polygonal support", IEEE Trans. Ann. Prop.  39, 1441 (1991)
"""

from GenericBackend import genericBackend as gb
from scipy.special import jn
from inkstone.ft.poly_area import poly_area


def ft_2d_poly_1(vertices, ks):
    """
    Calculate the fourier transform of a function with value 1 inside a polygon and 0 outside. Assuming none of the required k is at (0, 0)

    Parameters
    ----------
    vertices    :   list[tuple[float, float]] or ndarray
                    vertices of the polygon, in counterclockwise sequence.
                    Can give a list of the coordinates, each vertex is (x, y) tuple. Or, give a in nx2 shaped ndarray
    ks          :   list[tuple[float, float]] or ndarray
                    a list of k vectors. The k vectors can't be (0, 0)
                    can give a list of k's, each k is a (kx, ky) tuple. Or, give a mx2 shaped ndarray

    Returns
    -------
    s           :   list[complex]
                    1d array, value of Fourier coefficient at the corresponding ks points.

    """
    # convert to array
    ksa = gb.parseData(ks)  # each row is a [kx, ky]

    ks_nm = gb.la.norm(ksa, axis=-1)
    coef = -1j / ks_nm ** 2

    # convert tuples to numpy arrays
    verti = gb.parseData(vertices)

    # generate a shifted array of vertices of [r2, r3, ..., rn, r1], if originally [r1, r2, ..., rn]
    verti_roll = gb.roll(verti, -1, axis=0)

    rnc = (verti + verti_roll) / 2.

    ln = verti_roll - verti

    cross = gb.cross(ln[:, None, :], ksa[None, :, :])
    # Say ln.shape is mx2, ksa is nx2, this gives mxn shape

    term1 = cross * (gb.exp(-1j * rnc @ ksa.T)) * jn(0, ln @ ksa.T / 2.)

    s = coef * gb.sum(term1, axis=0)

    return s.tolist()


def ft_2d_poly(vertices, ks,gb=gb):
    """
    Calculate the Fourier transform of a function with value 1 inside a polygon shape and 0 outside.

    Parameters
    ----------
    vertices    :   list[tuple[float, float]]
                    vertices of the polygon, in counterclockwise sequence.
    ks          :   list[tuple[float, float]]
                    a list of k vectors.

    Returns
    -------
    s           :   list[complex]
                    1d array, value of Fourier coefficient at the corresponding ks points.

    Notes
    -----
    For k == (0, 0) and k != (0, 0), call different subroutine.
    """
    ksa = gb.parseData(ks)  # convert to array
    ks_nm = gb.la.norm(ksa, axis=-1)  # calculate the norm of each k vector
    idx_0 = gb.where(ks_nm == 0)[0]  # index to where k is (0, 0)
    idx_i = gb.where(ks_nm != 0)[0]  # index to where k is not (0, 0)
    ksa1 = gb.delete(ksa, idx_0, axis=0)  # new ks array that doesn't contain (0, 0)

    s1 = gb.parseData(ft_2d_poly_1(vertices, ksa1))
    a = poly_area(vertices)

    s = 1j * gb.zeros(ks_nm.size)
    s[idx_i] = s1
    s[idx_0] = a

    return s.tolist()

