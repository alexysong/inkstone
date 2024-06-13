# -*- coding: utf-8 -*-

from GenericBackend import genericBackend as gb


def poly_area(vertices,gb=gb):
    """

    Parameters
    ----------
    vertices        :   list[tuple(float, float)]
                        the vertices of the polygon ccw order

    Returns
    -------
    a               :   float
                        area of polygon
    """
    verti = gb.parseData(vertices)
    x = verti[:, 0]
    y = verti[:, 1]

    a = 0.5 * gb.abs(gb.dot(x, gb.roll(y, 1)) - gb.dot(y, gb.roll(x, 1)))

    return a
