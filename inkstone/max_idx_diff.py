# -*- coding: utf-8 -*-

import numpy as np


def max_idx_diff(idx):
    """
    imagine a 2d grid. given a list of integer indices (tuples), calculate the maximum relative difference in x and y in these indices.

    Parameters
    ----------
    idx     :   list[tuple[int, int]]

    Returns
    -------
    m, n    :   int

    """

    idxa = np.array(idx)  # (N， 2) shape array
    m = idxa[:, 0].max() - idxa[:, 0].min()  # max m index as in g = mb1 + nb2
    n = idxa[:, 1].max() - idxa[:, 1].min()

    return m, n

