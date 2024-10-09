# -*- coding: utf-8 -*-

import inkstone.backends.BackendLoader as bl


def max_idx_diff(idx,gb=bl.backend()):
    """
    imagine a 2d grid. given a list of integer indices (tuples), calculate the maximum relative difference in x and y in these indices.

    Parameters
    ----------
    idx     :   list[tuple[int, int]]

    Returns
    -------
    m, n    :   int

    """
    # wcai: none checking
    if not idx:
        raise ValueError("Received None/Empty input")
    
    idxa = gb.data(idx)  # (N， 2) shape array
    m = idxa[:, 0].max() - idxa[:, 0].min()  # max m index as in g = mb1 + nb2
    n = idxa[:, 1].max() - idxa[:, 1].min()

    return m, n

