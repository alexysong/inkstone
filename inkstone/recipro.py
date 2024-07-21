# -*- coding: utf-8 -*-

from inkstone.backends.BackendLoader import bg


def recipro(a1, a2, gb=bg.backend):
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

    a1n, a2n = [gb.la.norm(gb.parseData(a, dtype=gb.float64)) for a in [a1, a2]]

    if a1n == 0.:
        if a2n == 0.:
            raise Exception("The two lattice vectors can't be both zero vectors.")
        else:
            b1 = gb.parseData([float('inf'), float('inf')])
            b2 = 2 * gb.pi / (a2n ** 2) * a2
    else:
        if a2n == 0.:
            b1 = 2 * gb.pi / (a1n ** 2) * a1
            b2 = gb.parseData([float('inf'), float('inf')])
        else:
            ar = gb.abs(gb.cross(a1, a2))  # area
            coef = 2 * gb.pi / ar
            
            b1 = gb.parseList([coef * a2[1], -coef * a2[0]])
            b2 = gb.parseList([-coef * a1[1], coef * a1[0]])
    return b1,b2


