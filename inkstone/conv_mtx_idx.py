# -*- coding: utf-8 -*-

import numpy as np


def conv_mtx_idx_2d(idx1, idx2):
    """
    Indexing matrix to generate epsilon and mu convolution matrices.

    Parameters
    ----------
    idx1, idx2      :   list[tuple[int, int]]

    Returns
    -------
    cmi             :   ndarray
                        convolution matrix index, shape NxMx2
    """

    i1 = np.array(idx1)  # (N, 2) shape
    i2 = np.array(idx2)

    cmi = i1[:, None, :] - i2[None, :, :]  # (M, N, 2) shape, each element of (M, N, ...) is the 2 indices for mtx

    return cmi


