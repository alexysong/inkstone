# -*- coding: utf-8 -*-

from GenericBackend import genericBackend as gb
import scipy.linalg as sla
import time
# import scipy.sparse as sps
# import warnings


def rsp(sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22,gb=gb):
    """
    Take the Redheffer star product (rsp) of two scattering matrices.

    Parameters
    ----------
    sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22      :   ndarray
                                                            scattering matrix A and B

    Returns
    -------
    s11     :   ndarray
    s12     :   ndarray
    s21     :   ndarray
    s22     :   ndarray

    References
    ----------

    UTEP EMLab.
    Victor's Notes on Redheffer star product is actually skewed.
    """
    time1 = time.process_time()

    # identity matrix
    # idt = gb.diag(gb.ones(sa11.shape[0]))
    idt = gb.eye(sa11.shape[0], dtype=gb.complex128)

    # UTEP CEM (correct)
    # (bi, ao) = S (ai, bo), i, o means left 'input' region and right 'output' region, a means right going, b means left going
    # iv12 = la.inv(idt - sb11 @ sa22)
    # iv21 = la.inv(idt - sa22 @ sb11)
    #
    # s11 = sa11 + sa12 @ iv12 @ sb11 @ sa21
    # s12 = sa12 @ iv12 @ sb12
    # s21 = sb21 @ iv21 @ sa21
    # s22 = sb22 + sb21 @ iv21 @ sa22 @ sb12

    t1 = idt - sb11 @ sa22
    t2 = idt - sa22 @ sb11
    p1 = sla.solve(t1.T, sa12.T).T
    p2 = sla.solve(t2.T, sb21.T).T

    s11 = sa11 + p1 @ sb11 @ sa21
    s12 = p1 @ sb12
    s21 = p2 @ sa21
    s22 = sb22 + p2 @ sa22 @ sb12

    # Victor's, and several other online notes.
    # This is skewed, i.e. s11 is transmission, s12 is reflection.
    # i.e. (ao, bi) = S (ai, bo)
    # s21 = sa21 + sa22 @ la.inv(idt - sb21 @ sa12) @ sb21 @ sa11
    # s22 = sa22 @ la.inv(idt - sb21 @ sa12) @ sb22
    # s11 = sb11 @ la.inv(idt - sa12 @ sb21) @ sa11
    # s12 = sb12 + sb11 @ la.inv(idt - sa12 @ sb21) @ sa12 @ sb22

    # print('RSP', time.process_time() - time1)

    return s11, s12, s21, s22


def rsp_sa21Tlu(sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22):
    """
    Redheffer star product (rsp) of two scattering matrices, assuming sa21 is special.

    Parameters
    ----------
    sa11    :   any
    sa12    :   any
    sa21    :   Tuple[any, any]
                Assuming sa21 = A^-1, then this is the LU decomposition factor of A^T.
    sa22    :   any
    sb11    :   any
    sb12    :   any
    sb21    :   any
    sb22    :   any

    Returns
    -------
    s11     :   any
    s12     :   any
    s21     :   any
    s22     :   any

    """
    idt = gb.eye(sa11.shape[0], dtype=gb.complex128)

    t1 = idt - sb11 @ sa22
    t2 = idt - sa22 @ sb11
    p1 = sla.solve(t1.T, sa12.T).T
    p2 = sla.solve(t2.T, sb21.T).T

    s11 = sa11 + p1 @ sla.lu_solve(sa21, sb11.T).T
    s12 = p1 @ sb12
    s21 = sla.lu_solve(sa21, p2.T).T
    s22 = sb22 + p2 @ sa22 @ sb12

    return s11, s12, s21, s22


def rsp_sa12lu(sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22):
    """
    Redheffer star product (rsp) of two scattering matrices, assuming sa12 is special.

    Parameters
    ----------
    sa11    :   any
    sa12    :   Tuple[any, any]
                Assuming sa12 = A^-1, then this is the LU decomposition factor of A.
    sa21    :   any
    sa22    :   any
    sb11    :   any
    sb12    :   any
    sb21    :   any
    sb22    :   any

    Returns
    -------
    s11     :   any
    s12     :   any
    s21     :   any
    s22     :   any

    """
    idt = gb.eye(sa11.shape[0], dtype=gb.complex128)

    t1 = idt - sb11 @ sa22
    t2 = idt - sa22 @ sb11

    p2 = sla.solve(t2.T, sb21.T).T

    s11 = sa11 + sla.lu_solve(sa12, (sla.solve(t1, sb11) @ sa21))
    s12 = sla.lu_solve(sa12, (sla.solve(t1, sb12)))
    s21 = p2 @ sa21
    s22 = sb22 + p2 @ sa22 @ sb12

    return s11, s12, s21, s22


def rsp_sb12Tlu(sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22):
    """
    Redheffer star product (rsp) of two scattering matrices, assuming sb12 is special.

    Parameters
    ----------
    sa11    :   any
    sa12    :   any
    sa21    :   any
    sa22    :   any
    sb11    :   any
    sb12    :   Tuple[any, any]
                assuming sb12 = A^-1, then this is the LU decomposition of A^T.
    sb21    :   any
    sb22    :   any

    Returns
    -------
    s11     :   any
    s12     :   any
    s21     :   any
    s22     :   any

    """
    idt = gb.eye(sa11.shape[0], dtype=gb.complex128)

    t1 = idt - sb11 @ sa22
    t2 = idt - sa22 @ sb11
    p1 = sla.solve(t1.T, sa12.T).T
    p2 = sla.solve(t2.T, sb21.T).T

    s11 = sa11 + p1 @ sb11 @ sa21
    s12 = sla.lu_solve(sb12, p1.T).T
    s21 = p2 @ sa21
    s22 = sb22 + p2 @ sla.lu_solve(sb12, sa22.T).T

    return s11, s12, s21, s22


def rsp_sb21lu(sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22):
    """
    Redheffer star product (rsp) of two scattering matrices, assuming sb21 is special.

    Parameters
    ----------
    sa11    :   any
    sa12    :   any
    sa21    :   any
    sa22    :   any
    sb11    :   any
    sb12    :   any
    sb21    :   Tuple[any, any]
                assuming sb12 = A^-1, then this is the LU decomposition of A.
    sb22    :   any

    Returns
    -------
    s11     :   any
    s12     :   any
    s21     :   any
    s22     :   any

    """
    idt = gb.eye(sa11.shape[0], dtype=gb.complex128)

    t1 = idt - sb11 @ sa22
    t2 = idt - sa22 @ sb11
    p1 = sla.solve(t1.T, sa12.T).T
    # p2 = sla.solve(t2.T, sb21.T).T

    s11 = sa11 + p1 @ sb11 @ sa21
    s12 = p1 @ sb12
    s21 = sla.lu_solve(sb21, sla.solve(t2, sa21))
    s22 = sb22 + sla.lu_solve(sb21, sla.solve(t2, sa22) @ sb12)

    return s11, s12, s21, s22


def rsp_sa12lu_sb21lu(sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22):
    """
    Redheffer star product (rsp) of two scattering matrices, assuming sb21 is special.

    Parameters
    ----------
    sa11    :   any
    sa12    :   Tuple[any, any]
                assuming sa12 = A^-1, then this is the LU decomposition of A.
    sa21    :   any
    sa22    :   any
    sb11    :   any
    sb12    :   any
    sb21    :   Tuple[any, any]
                assuming sb12 = A^-1, then this is the LU decomposition of A.
    sb22    :   any

    Returns
    -------
    s11     :   any
    s12     :   any
    s21     :   any
    s22     :   any

    """
    idt = gb.eye(sa11.shape[0], dtype=gb.complex128)

    t1 = idt - sb11 @ sa22
    t2 = idt - sa22 @ sb11
    # p1 = sla.solve(t1.T, sa12.T).T
    # p2 = sla.solve(t2.T, sb21.T).T

    s11 = sa11 + sla.lu_solve(sa12, sla.solve(t1, sb11)) @ sa21
    s12 = sla.lu_solve(sa12, sla.solve(t1, sb12))
    s21 = sla.lu_solve(sb21, sla.solve(t2, sa21))
    s22 = sb22 + sla.lu_solve(sb21, sla.solve(t2, sa22) @ sb12)

    return s11, s12, s21, s22


def rsp_sa21Tlu_sb21lu(sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22):
    """
    attn: this method is not fully tested

    Redheffer star product (rsp) of two scattering matrices, assuming sb21 is special.

    Parameters
    ----------
    sa11    :   any
    sa12    :   any
    sa21    :   Tuple[any, any]
                assuming sa12 = A^-1, then this is the LU decomposition of A^T.
    sa22    :   any
    sb11    :   any
    sb12    :   any
    sb21    :   Tuple[any, any]
                assuming sb12 = A^-1, then this is the LU decomposition of A.
    sb22    :   any

    Returns
    -------
    s11     :   any
    s12     :   any
    s21     :   any
    s22     :   any

    """
    idt = gb.eye(sa11.shape[0], dtype=gb.complex128)

    t1 = idt - sb11 @ sa22
    t2 = idt - sa22 @ sb11
    # p1 = sla.solve(t1.T, sa12.T).T
    # p2 = sla.solve(t2.T, sb21.T).T

    s11 = sa11 + sa12 @ sla.lu_solve(sa21, sla.solve(t1, sb11).T).T
    s12 = sa12 @ sla.solve(t1, sb12)
    s21 = sla.lu_solve(sb21, sla.solve(t2, sla.lu_solve(sa21, idt).T))
    s22 = sb22 + sla.lu_solve(sb21, sla.solve(t2, sa22) @ sb12)

    return s11, s12, s21, s22


def rsp_sa12lu_sb12Tlu(sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22):
    """
    attn: this method is not fully tested

    Redheffer star product (rsp) of two scattering matrices, assuming sb21 is special.

    Parameters
    ----------
    sa11    :   any
    sa12    :   Tuple[any, any]
                assuming sa12 = A^-1, then this is the LU decomposition of A.
    sa21    :   any
    sa22    :   any
    sb11    :   any
    sb12    :   Tuple[any, any]
                assuming sb12 = A^-1, then this is the LU decomposition of A.T
    sb21    :   any
    sb22    :   any

    Returns
    -------
    s11     :   any
    s12     :   any
    s21     :   any
    s22     :   any

    """
    idt = gb.eye(sa11.shape[0], dtype=gb.complex128)

    t1 = idt - sb11 @ sa22
    t2 = idt - sa22 @ sb11
    # p1 = sla.solve(t1.T, sa12.T).T
    # p2 = sla.solve(t2.T, sb21.T).T

    s11 = sa11 + sla.lu_solve(sa12, sla.solve(t1, sb11)) @ sa21
    s12 = sla.lu_solve(sa12, sla.solve(t1, sla.lu_solve(sb12, idt).T))
    s21 = sb21 @ sla.solve(t2, sa21)
    s22 = sb22 + sb21 @ sla.lu_solve(sb12, sla.solve(t2, sa22).T).T

    return s11, s12, s21, s22

