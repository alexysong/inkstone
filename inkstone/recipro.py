# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la


def recipro(a1, a2):
    """
    given two lattice vectors, give two reciprocal lattice vectors
    If one of the lattice vectors is zero, then the returned corresponding reciprocal lattice vector is float('inf').

    Parameters
    ----------
    a1  :   tuple[float, float]
    a2  :   tuple[float, float]

    Returns
    -------
    b1  :   tuple[float, float]
    b2  :   tuple[float, float]

    """

    a1, a2 = [np.array(a) for a in [a1, a2]]
    a1n, a2n = [la.norm(a) for a in [a1, a2]]

    if a1n == 0.:
        if a2n == 0.:
            raise Exception("The two lattice vectors can't be both zero vectors.")
        else:
            b1 = (float('inf'), float('inf'))
            b2 = tuple(2 * np.pi / (a2n ** 2) * a2)
    else:
        if a2n == 0.:
            b1 = tuple(2 * np.pi / (a1n ** 2) * a1)
            b2 = (float('inf'), float('inf'))
        else:
            ar = np.abs(np.cross(a1, a2))  # area
            coef = 2 * np.pi / ar

            b1 = (coef * a2[1], -coef * a2[0])
            b2 = (-coef * a1[1], coef * a1[0])

    return b1, b2


