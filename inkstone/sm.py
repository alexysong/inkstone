# -*- coding: utf-8 -*-
import inkstone.backends.BackendLoader as bl
# import scipy.linalg as sla
# import scipy.sparse as sps
from .rsp import rsp_sa21Tlu, rsp_sb12Tlu

gb = bl.backend()
def s_1l(thickness, ql, al0, bl0):
    """
    calculate the scattering matrix of 1 layer.

    Parameters
    ----------
    thickness   :   float
                    thickness of layer
    ql          :   any
                    1d array, eigen propagation constant in z in layer
    al0         :   any
    bl0         :   any
                    interface matrix

    Returns
    -------
    s11, s12, s21, s22  :   any
                            four elements of the scattering matrix

    """

    a = al0
    b = bl0
    # alu = gb.lu_factor(a)
    aTlu = gb.lu_factor(a.T)
    ba_ = (gb.solve(a.T, b.T)).T

    fl = gb.exp(1j * ql * thickness)

    # a_fbafb = a - fl[:, None] * b @ gb.lu_solve(alu, (fl[:, None] * b))
    a_fbafb = a - fl[:, None] * ba_ @ (fl[:, None] * b)
    a_fbafb_lu = gb.lu_factor(a_fbafb)

    # fbafa_b = (fl[:, None] * b @ gb.lu_solve(alu, fl[:, None] * a) - b)
    fbafa_b = fl[:, None] * ba_ @ (fl[:, None] * a) - b
    # fa_bab = (fl[:, None] * (a - b @ gb.lu_solve(alu, b)))
    fa_bab = (fl[:, None] * (a - ba_ @ b))

    s11 = gb.lu_solve(a_fbafb_lu, fbafa_b)
    s12 = gb.lu_solve(a_fbafb_lu, fa_bab)

    # # debugging
    # d = fl - 1.
    # db = d[:, None] * b
    # dbab = d[:, None] * ba_ @ b
    # badb = ba_ @ (d[:, None] * b)
    # dbadb = d[:, None] * ba_ @ ((d[:, None] * b))

    s22 = s11
    s21 = s12

    return s11, s12, s21, s22


def s_1l_rsp(thickness: float,
             ql: any,
             afl: any,
             bfl: any):
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

    f = gb.exp(1j * ql * thickness)

    # ia = la.inv(a)
    aTlu = gb.lu_factor(a.T)
    aTlu2 = (gb.clone(aTlu[0]), gb.clone(aTlu[1]))
    a1 = aTlu2[0]
    a1 = gb.assignAndMultiply(a1, gb.triu_indices(a1.shape[0]), 0.5)
    alu = gb.lu_factor(a)
    # alu2 = (alu[0].copy(), alu[1].copy())
    # a1 = alu2[0]
    # a1[gb.triu_indices(a1.shape[0])] *= 0.5

    ab = gb.lu_solve(alu, b)
    ba = gb.lu_solve(aTlu, b.T).T
    bfa = gb.lu_solve(aTlu, (b*f).T).T

    bab = b @ ab
    a_bab = a - bab
    a_babf = a_bab * f
    a_bfabf = a - b * f @ ab * f
    # try:
    term = gb.solve(a_bfabf.T, a_babf.T).T
    # except Exception as e:
    #     if (ql == 0.).any():
    #         warn('Singular matrix encountered. Layer eigen propagation constant detected.', RuntimeWarning)
    #         print(la.cond(a_bfabf))
    #         print(la.cond(a_babf))
    #         term = gb.solve((a_bfabf+1e-14 * gb.eye(len(ql))).T, (a_babf+1e-14*gb.eye(len(ql))).T).T
    #         print(la.cond(term))
    #     else:
    #         return e

    s11 = ba - term @ bfa
    s12 = term
    s21 = s12
    s22 = s11

    # alu = gb.lu_factor(a)
    # aib = gb.lu_solve(alu, b)
    # sl11 = b @ ia
    sl11 = gb.lu_solve(aTlu, b.T).T
    # sl12 = a - b @ ia @ b
    sl12 = 0.5 * (a - b @ ab)
    sl21 = aTlu2
    sl22 = -ab

    # # debugging
    # ts = gb.ones(tng, dtype=complex)
    # ts[2] *= 1e7
    # ts[4] *= 1e9
    #
    # term_ts = gb.solve((a_bfabf*ts).T, (a_babf*ts).T).T
    # diff = term - term_ts
    # diffmag = gb.abs(diff).max()
    # # end of debugging


    # # explicit rsp subroutine
    # sf11 = gb.zeros((tng, tng))
    # sf12 = gb.diag(f)
    # sf21 = gb.diag(f)
    # sf22 = gb.zeros((tng, tng))
    #
    # sr11 = sl22
    # sr12 = sl21
    # sr21 = sl12
    # sr22 = sl11
    #
    # ss = rsp_sa21Tlu(sl11, sl12, sl21, sl22, sf11, sf12, sf21, sf22)
    # sm = rsp_sb12Tlu(*ss, sr11, sr12, sr21, sr22)

    # # debugging
    # f = gb.exp(1j * ql * 0.35)
    # sngl1 = (a - b @ ab)*f
    # sngl2 = a - b * f @ ab * f
    # prdc = gb.solve(sngl2.T, sngl1.T).T
    # prdc1 = prdc.copy()
    # prdc1[gb.where(gb.abs(prdc) < 1e-14)] = 0.
    # prdctest = gb.solve(sngl1.T, sngl1.T).T

    # # debugging
    # # wong left/right scattering matrix
    # f = gb.exp(1j * ql * 0.5)
    # sngl1 = f * (a - b @ ab)
    # sngl2 = gb.solve(b.T, a).T - f * gb.solve(a.T, b).T * f
    # prdc = gb.solve(sngl2, sngl1)
    # prdc1 = prdc.copy()
    # prdc1[gb.where(gb.abs(prdc) < 1e-14)] = 0.
    # # prdctest = gb.solve(sngl1.T, sngl1.T).T

    return s11, s12, s21, s22


def s_1l_rsp_lrd(thickness: float,
             ql: any,
             afl: any,
             bfl: any,
             afl2: any,
             bfl2: any):
    """
    Calculate the scattering matrix of 1 layer, using Redheffer products, using interface matrices from outside to this layer. The layer is assumed to be sandwiched between two infinitely thin layer of some other materials.

    lrd: assuming left and right different

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
    aTlu = gb.lu_factor(a.T)
    aTlu2 = (aTlu[0].copy(), aTlu[1].copy())
    a1 = aTlu2[0]
    a1 = gb.assignAndMultiply(a1, gb.triu_indices(a1.shape[0]), 0.5)
    alu = gb.lu_factor(a)
    # alu2 = (alu[0].copy(), alu[1].copy())
    # a1 = alu2[0]
    # a1[gb.triu_indices(a1.shape[0])] *= 0.5

    ab = gb.lu_solve(alu, b)

    # alu = gb.lu_factor(a)
    # aib = gb.lu_solve(alu, b)
    # sl11 = b @ ia
    sl11 = gb.lu_solve(aTlu, b.T).T
    # sl12 = a - b @ ia @ b
    sl12 = 0.5 * (a - b @ ab)
    sl21 = aTlu2
    sl22 = -ab

    f = gb.exp(1j * ql * thickness)

    sf11 = gb.zeros((tng, tng))
    sf12 = gb.diag(f)
    sf21 = gb.diag(f)
    sf22 = gb.zeros((tng, tng))

    a2 = afl2
    b2 = bfl2

    # ia = la.inv(a)
    a2Tlu = gb.lu_factor(a2.T)
    a2Tlu2 = (gb.clone(a2Tlu[0]), gb.clone(a2Tlu[1]))
    a1 = a2Tlu2[0]
    a1 = gb.assignAndMultiply(a1, gb.triu_indices(a1.shape[0]), 0.5)
    a2lu = gb.lu_factor(a2)
    # alu2 = (alu[0].copy(), alu[1].copy())
    # a1 = alu2[0]
    # a1[gb.triu_indices(a1.shape[0])] *= 0.5

    a2b2 = gb.lu_solve(a2lu, b2)

    sr11 = -a2b2
    sr12 = a2Tlu2
    sr21 = 0.5 * (a2 - b2 @ a2b2)
    sr22 = gb.lu_solve(a2Tlu, b2.T).T

    # # debugging
    # f = gb.exp(1j * ql * 0.35)
    # sngl1 = (a - b @ ab)*f
    # sngl2 = a - b * f @ ab * f
    # prdc = gb.solve(sngl2.T, sngl1.T).T
    # prdc1 = prdc.copy()
    # prdc1[gb.where(gb.abs(prdc) < 1e-14)] = 0.
    # prdctest = gb.solve(sngl1.T, sngl1.T).T

    # # debugging
    # # wong left/right scattering matrix
    # f = gb.exp(1j * ql * 0.5)
    # sngl1 = f * (a - b @ ab)
    # sngl2 = gb.solve(b.T, a).T - f * gb.solve(a.T, b).T * f
    # prdc = gb.solve(sngl2, sngl1)
    # prdc1 = prdc.copy()
    # prdc1[gb.where(gb.abs(prdc) < 1e-14)] = 0.
    # # prdctest = gb.solve(sngl1.T, sngl1.T).T

    ss = rsp_sa21Tlu(sl11, sl12, sl21, sl22, sf11, sf12, sf21, sf22)
    sm = rsp_sb12Tlu(*ss, sr11, sr12, sr21, sr22)

    return sm


def s_1l_1212(a, b):
    """
    Calculate the scattering matrix of material 1 to 2 using interface matrices defined from 1 to 2.

    Parameters
    ----------
    a     :   any
    b     :   any

    Returns
    -------
    s11  :  any
    s12  :  any
    s21  :  Tuple[any, any]
            lu factorization
    s22  :  any
    """

    a = a
    b = b

    aTlu = gb.lu_factor(a.T)
    aTlu2 = (gb.clone(aTlu[0]), gb.clone(aTlu[1]))
    aT1 = aTlu2[0]
    a1 = gb.assignAndMultiply(aT1, gb.triu_indices(aT1.shape[0]), 0.5)
    alu = gb.lu_factor(a)
    alu2 = (gb.clone(alu[0]), gb.clone(alu[1]))
    a1 = alu2[0]
    a1 = gb.assignAndMultiply(a1, gb.triu_indices(a1.shape[0]), 0.5)
    ab = gb.lu_solve(alu, b)

    s11 = gb.lu_solve(aTlu, b.T).T
    s12 = 1./2. * (a - b @ ab)
    s21 = alu2
    s22 = - ab

    return s11, s12, s21, s22


def s_1l_1221(a, b):
    """
    Calculate the scattering matrix of the interface from material 1 to 2 using interface matrices defined from 2 to 1.

    Parameters
    ----------
    a     :   any
    b     :   any

    Returns
    -------
    s11  :  any
    s12  :  Tuple[any, any]
            lu factorization
    s21  :  any
    s22  :  any
    """

    a = a
    b = b

    alu = gb.lu_factor(a)
    alu2 = (gb.clone(alu[0]), gb.clone(alu[1]))
    a1 = alu2[0]
    a1 = gb.assignAndMultiply(a1, gb.triu_indices(a1.shape[0]), 0.5)
    aTlu = gb.lu_factor(a.T)
    aTlu2 = (gb.clone(aTlu[0]), gb.clone(aTlu[1]))
    at1 = aTlu2[0]
    at1 = gb.assignAndMultiply(at1, gb.triu_indices(at1.shape[0]), 0.5)
    ab = gb.lu_solve(alu, b)

    s11 = - ab
    s12 = alu2
    s21 = 1./2. * (a - b @ ab)
    s22 = gb.lu_solve(aTlu, b.T).T

    return s11, s12, s21, s22


