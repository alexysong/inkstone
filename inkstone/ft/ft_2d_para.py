# -*- coding: utf-8 -*-

from GenericBackend import genericBackend as gb
from inkstone.ft.ft_1d_sq import ft_1d_sq


def ft_2d_para(a, b, ks, center=(0, 0), shear_angle=90., rotate_angle=0.,gb=gb):
    """
    Calculate the fourier transform of a function that is 1 in side a parallelogram and 0 outside.

    The parallelogram has actual side lengths of a and b (b is on the ccw side of a).

    The parallelogram is rotated by `angle` and then shifted to `center`.

    Parameters
    ----------
    a               :   float
    b               :   float
                        side lengths of the parallelogram
    ks              :   list[tuple[float, float]]
    center          :   tuple[float, float]
    shear_angle     :   float
                        angle between a and b (a ccw rotates this angle to b)
    rotate_angle    :   float
                        ccw rotation angle of the whole shape in degrees

    Returns
    -------
    s               :   list[complex]
                        1d array
    """

    ia = gb.pi * shear_angle / 180.
    b1 = gb.sin(ia) * b
    m = gb.tan(gb.pi/2 - ia)

    shear = gb.inpurParser([[1, m],
                      [0, 1]])

    ksa = gb.inpurParser(ks)  # Nx2 array

    cen = gb.inpurParser(center)

    ang = gb.pi * rotate_angle / 180.

    rot = gb.inpurParser([[gb.cos(ang), -gb.sin(ang)], [gb.sin(ang), gb.cos(ang)]])

    A = rot @ shear

    aksa = (A.T @ ksa.T).T

    sx = ft_1d_sq(a, aksa[:, 0])
    sy = ft_1d_sq(b1, aksa[:, 1])

    s = gb.exp(-1j * cen @ ksa.T) * sx * sy

    return s.tolist()

