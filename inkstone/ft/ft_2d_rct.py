# -*- coding: utf-8 -*-

import numpy as np
from inkstone.ft.ft_1d_sq import ft_1d_sq


def ft_2d_rct(a, b, ks, center=None, angle=0.):
    """

    Parameters
    ----------
    a, b            :   float
                        side lengths in x and y direction, before rotation
    ks              :   list[tuple[float, float]]
    center          :   tuple[float, float]
    angle           :   float
                        ccw rotation angle in degrees

    Returns
    -------
    s               :   list[complex]
                        1d array
    """

    if center is None:
        center = (0., 0.)
    cen = np.array(center)

    ksa = np.array(ks)  # Nx2 array

    ang = np.pi * angle / 180.
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    aksa = (rot.T @ ksa.T).T

    sx = ft_1d_sq(a, aksa[:, 0])
    sy = ft_1d_sq(b, aksa[:, 1])

    s = np.exp(-1j * cen @ ksa.T) * sx * sy

    return s.tolist()

