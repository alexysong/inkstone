# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
from scipy.special import jn


def ft_2d_ellip(a, b, ks, center=None, angle=0.):
    """
    Calculate the fourier transform of a function with value 1 inside a ellipse and 0 outside.

    The ellipse has half widths a and b, rotated ccw of `angle`, then shifted to `center`.

    Parameters
    ----------
    a, b        :   float
                    half widths of the ellipse, assume a in x direction and b in y
    ks          :   list[tuple[float, float]]
                    list of (kx, ky) points
    center      :   tuple[float, float]
                    the center of the disk
    angle       :   float
                    the rotation angle of the ellipse, ccw, in degrees

    Returns
    -------
    s       :   list[complex]
                1d array, Fourier coefficient at the input ks positions

    """

    ang = angle / 180. * np.pi
    stretch = np.array([[a, 0], [0, b]])
    rotate = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    aff = rotate @ stretch

    ksa = np.array(ks)  # nx2 shape
    aksa = (aff.T @ ksa.T).T
    aks_nm = np.linalg.norm(aksa, axis=-1)  # 1d array of n. The norm of each k vector
    idx_0 = np.where(aks_nm == 0)[0]  # index to where k is (0, 0)
    idx_i = np.where(aks_nm != 0)[0]  # index to where k is not (0, 0)
    aks_nm1 = np.delete(aks_nm, idx_0)

    ksa1 = np.delete(ksa, idx_0, axis=0)  # new ks array that doesn't contain (0, 0)

    if center is None:
        center = (0, 0)
    cent = np.array(center)

    s = 1j * np.zeros(aks_nm.size)
    s[idx_i] = np.abs(la.det(aff)) * 2 * np.pi * jn(1, aks_nm1) / aks_nm1 * np.exp(-1j * cent @ ksa1.T)
    s[idx_0] = np.pi * a * b

    return s.tolist()

