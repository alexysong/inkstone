# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import time
# import scipy.sparse as sps
# import warnings


def rsp(sa11, sa12, sa21, sa22, sb11, sb12, sb21, sb22):
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
    # idt = np.diag(np.ones(sa11.shape[0]))
    idt = np.eye(sa11.shape[0], dtype=complex)

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
    sa11    :   np.ndarray
    sa12    :   np.ndarray
    sa21    :   Tuple[np.ndarray, np.ndarray]
                Assuming sa21 = A^-1, then this is the LU decomposition factor of A^T.
    sa22    :   np.ndarray
    sb11    :   np.ndarray
    sb12    :   np.ndarray
    sb21    :   np.ndarray
    sb22    :   np.ndarray

    Returns
    -------
    s11     :   np.ndarray
    s12     :   np.ndarray
    s21     :   np.ndarray
    s22     :   np.ndarray

    """
    idt = np.eye(sa11.shape[0], dtype=complex)

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
    sa11    :   np.ndarray
    sa12    :   Tuple[np.ndarray, np.ndarray]
                Assuming sa12 = A^-1, then this is the LU decomposition factor of A.
    sa21    :   np.ndarray
    sa22    :   np.ndarray
    sb11    :   np.ndarray
    sb12    :   np.ndarray
    sb21    :   np.ndarray
    sb22    :   np.ndarray

    Returns
    -------
    s11     :   np.ndarray
    s12     :   np.ndarray
    s21     :   np.ndarray
    s22     :   np.ndarray

    """
    idt = np.eye(sa11.shape[0], dtype=complex)

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
    sa11    :   np.ndarray
    sa12    :   np.ndarray
    sa21    :   np.ndarray
    sa22    :   np.ndarray
    sb11    :   np.ndarray
    sb12    :   Tuple[np.ndarray, np.ndarray]
                assuming sb12 = A^-1, then this is the LU decomposition of A^T.
    sb21    :   np.ndarray
    sb22    :   np.ndarray

    Returns
    -------
    s11     :   np.ndarray
    s12     :   np.ndarray
    s21     :   np.ndarray
    s22     :   np.ndarray

    """
    idt = np.eye(sa11.shape[0], dtype=complex)

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
    sa11    :   np.ndarray
    sa12    :   np.ndarray
    sa21    :   np.ndarray
    sa22    :   np.ndarray
    sb11    :   np.ndarray
    sb12    :   np.ndarray
    sb21    :   Tuple[np.ndarray, np.ndarray]
                assuming sb12 = A^-1, then this is the LU decomposition of A.
    sb22    :   np.ndarray

    Returns
    -------
    s11     :   np.ndarray
    s12     :   np.ndarray
    s21     :   np.ndarray
    s22     :   np.ndarray

    """
    idt = np.eye(sa11.shape[0], dtype=complex)

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
    sa11    :   np.ndarray
    sa12    :   Tuple[np.ndarray, np.ndarray]
                assuming sa12 = A^-1, then this is the LU decomposition of A.
    sa21    :   np.ndarray
    sa22    :   np.ndarray
    sb11    :   np.ndarray
    sb12    :   np.ndarray
    sb21    :   Tuple[np.ndarray, np.ndarray]
                assuming sb12 = A^-1, then this is the LU decomposition of A.
    sb22    :   np.ndarray

    Returns
    -------
    s11     :   np.ndarray
    s12     :   np.ndarray
    s21     :   np.ndarray
    s22     :   np.ndarray

    """
    idt = np.eye(sa11.shape[0], dtype=complex)

    t1 = idt - sb11 @ sa22
    t2 = idt - sa22 @ sb11
    # p1 = sla.solve(t1.T, sa12.T).T
    # p2 = sla.solve(t2.T, sb21.T).T

    s11 = sa11 + sla.lu_solve(sa12, sla.solve(t1, sb11)) @ sa21
    s12 = sla.lu_solve(sa12, sla.solve(t1, sb12))
    s21 = sla.lu_solve(sb21, sla.solve(t2, sa21))
    s22 = sb22 + sla.lu_solve(sb21, sla.solve(t2, sa22) @ sb12)

    return s11, s12, s21, s22
