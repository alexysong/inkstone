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


def s_1l_rsp(thickness: float,
             ql: np.ndarray,
             afl: np.ndarray,
             bfl: np.ndarray):
    """
    Calculate the scattering matrix of 1 layer, using Redheffer products, using interface matrices from outside to this layer. The layer is assumed to be sandwiched between two infinitely thin layer of some other materials.

    Parameters
    ----------
    thickness : thickness of layer
    ql : 1d array, eigen propagation constant in z in layer
    afl : interface matrix
    bfl : interface matrix

    Returns
    -------

    """
    a = afl
    b = bfl

    tng = afl.shape[0]

    # ia = la.inv(a)
    aTlu = sla.lu_factor(a.T)
    aTlu2 = (aTlu[0].copy(), aTlu[1].copy())
    a1 = aTlu2[0]
    a1[np.triu_indices(a1.shape[0])] *= 0.5
    alu = sla.lu_factor(a)
    # alu2 = (alu[0].copy(), alu[1].copy())
    # a1 = alu2[0]
    # a1[np.triu_indices(a1.shape[0])] *= 0.5

    ab = sla.lu_solve(alu, b)

    # alu = sla.lu_factor(a)
    # aib = sla.lu_solve(alu, b)
    # sl11 = b @ ia
    sl11 = sla.lu_solve(aTlu, b.T).T
    # sl12 = a - b @ ia @ b
    sl12 = 0.5 * (a - b @ ab)
    sl21 = aTlu2
    sl22 = -ab

    f = np.exp(1j * ql * thickness)

    sf11 = np.zeros((tng, tng))
    sf12 = np.diag(f)
    sf21 = np.diag(f)
    sf22 = np.zeros((tng, tng))

    sr11 = sl22
    sr12 = sl21
    sr21 = sl12
    sr22 = sl11

    from .rsp import rsp, rsp_sa21Tlu, rsp_sb12Tlu

    # # debugging
    # f = np.exp(1j * ql * 0.35)
    # sngl1 = (a - b @ ab)*f
    # sngl2 = a - b * f @ ab * f
    # prdc = sla.solve(sngl2.T, sngl1.T).T
    # prdc1 = prdc.copy()
    # prdc1[np.where(np.abs(prdc) < 1e-14)] = 0.
    # prdctest = sla.solve(sngl1.T, sngl1.T).T

    # # debugging
    # # wong left/right scattering matrix
    # f = np.exp(1j * ql * 0.5)
    # sngl1 = f * (a - b @ ab)
    # sngl2 = sla.solve(b.T, a).T - f * sla.solve(a.T, b).T * f
    # prdc = sla.solve(sngl2, sngl1)
    # prdc1 = prdc.copy()
    # prdc1[np.where(np.abs(prdc) < 1e-14)] = 0.
    # # prdctest = sla.solve(sngl1.T, sngl1.T).T

    ss = rsp_sa21Tlu(sl11, sl12, sl21, sl22, sf11, sf12, sf21, sf22)
    sm = rsp_sb12Tlu(*ss, sr11, sr12, sr21, sr22)

    return sm


def s_1l_1212(a, b):
    """
    Calculate the scattering matrix of material 1 to 2 using interface matrices defined from 1 to 2.

    Parameters
    ----------
    a     :   ndarray
    b     :   ndarray

    Returns
    -------
    s11  :  ndarray
    s12  :  ndarray
    s21  :  Tuple[np.ndarray, np.ndarray]
            lu factorization
    s22  :  ndarray
    """

    a = a
    b = b

    aTlu = sla.lu_factor(a.T)
    aTlu2 = (aTlu[0].copy(), aTlu[1].copy())
    aT1 = aTlu2[0]
    aT1[np.triu_indices(aT1.shape[0])] *= 0.5
    alu = sla.lu_factor(a)
    alu2 = (alu[0].copy(), alu[1].copy())
    a1 = alu2[0]
    a1[np.triu_indices(a1.shape[0])] *= 0.5
    ab = sla.lu_solve(alu, b)

    s11 = sla.lu_solve(aTlu, b.T).T
    s12 = 1./2. * (a - b @ ab)
    s21 = alu2
    s22 = - ab

    return s11, s12, s21, s22


def s_1l_1221(a, b):
    """
    Calculate the scattering matrix of the interface from material 1 to 2 using interface matrices defined from 2 to 1.

    Parameters
    ----------
    a     :   ndarray
    b     :   ndarray

    Returns
    -------
    s11  :  ndarray
    s12  :  Tuple[np.ndarray, np.ndarray]
            lu factorization
    s21  :  ndarray
    s22  :  ndarray
    """

    a = a
    b = b

    alu = sla.lu_factor(a)
    alu2 = (alu[0].copy(), alu[1].copy())
    a1 = alu2[0]
    a1[np.triu_indices(a1.shape[0])] *= 0.5
    aTlu = sla.lu_factor(a.T)
    aTlu2 = (aTlu[0].copy(), aTlu[1].copy())
    at1 = aTlu2[0]
    at1[np.triu_indices(at1.shape[0])] *= 0.5
    ab = sla.lu_solve(alu, b)

    s11 = - ab
    s12 = alu2
    s21 = 1./2. * (a - b @ ab)
    s22 = sla.lu_solve(aTlu, b.T).T

    return s11, s12, s21, s22


