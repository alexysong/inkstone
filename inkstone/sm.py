# -*- coding: utf-8 -*-

import numpy as np
# import numpy.linalg as la
import scipy.linalg as sla
# import scipy.sparse as sps
# import warnings


def s_1l(thickness, ql, al0, bl0):
    """
    calculate the scattering matrix of 1 layer.

    Parameters
    ----------
    thickness   :   float
                    thickness of layer
    ql          :   ndarray
                    1d array, eigen propagation constant in z in layer
    al0         :   ndarray
    bl0         :   ndarray
                    interface matrix

    Returns
    -------
    s11, s12, s21, s22  :   ndarray
                            four elements of the scattering matrix

    """

    a = al0
    b = bl0
    alu = sla.lu_factor(a)

    fl = np.exp(1j * ql * thickness)

    t1 = a - fl[:, None] * b @ sla.lu_solve(alu, (fl[:, None] * b))
    t1lu = sla.lu_factor(t1)

    s21 = sla.lu_solve(t1lu, (fl[:, None] * (a - b @ sla.lu_solve(alu, b))))
    s22 = sla.lu_solve(t1lu, (fl[:, None] * b @ sla.lu_solve(alu, fl[:, None] * a) - b))

    s11 = s22
    s12 = s21

    return s11, s12, s21, s22


def s_1l_in(al0, bl0):
    """
    Calculate the scattering matrix of the 'input' region.
    The input region may not be vacuum. It has a thickness of infinity. The s-matrix is defined for the interface from this infinite material to (thickness 0) vacuum.

    Parameters
    ----------
    al0     :   ndarray
    bl0     :   ndarray

    Returns
    -------
    s11, s12, s21, s22  :   ndarray
    """

    a = al0
    b = bl0

    aTlu = sla.lu_factor(a.T)
    aTlu2 = (aTlu[0].copy(), aTlu[1].copy())
    a1 = aTlu2[0]
    a1[np.triu_indices(a1.shape[0])] *= 0.5
    alu = sla.lu_factor(a)
    ab = sla.lu_solve(alu, b)

    s11 = sla.lu_solve(aTlu, b.T).T
    s12 = 1./2. * (a - b @ ab)
    s21 = aTlu2
    s22 = - ab

    return s11, s12, s21, s22


def s_1l_out(al0, bl0):
    """
    Calculate the scattering matrix of the 'output' region.
    The output region may not be vacuum. It has a thickness of infinity. The s-matrix is defined for the interface from (thickness 0) vacuum to this infinite material.

    Parameters
    ----------
    al0     :   ndarray
    bl0     :   ndarray

    Returns
    -------
    s11, s12, s21, s22  :   ndarray
    """

    a = al0
    b = bl0

    alu = sla.lu_factor(a)
    aTlu = sla.lu_factor(a.T)
    aTlu2 = (aTlu[0].copy(), aTlu[1].copy())
    a1 = aTlu2[0]
    a1[np.triu_indices(a1.shape[0])] *= 0.5
    ab = sla.lu_solve(alu, b)

    s11 = - ab
    s12 = aTlu2
    s21 = 1./2. * (a - b @ ab)
    s22 = sla.lu_solve(aTlu, b.T).T

    return s11, s12, s21, s22


