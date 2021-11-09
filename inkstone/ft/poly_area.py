# -*- coding: utf-8 -*-

import numpy as np


def poly_area(vertices):
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
    verti = np.array(vertices)
    x = verti[:, 0]
    y = verti[:, 1]

    a = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return a
