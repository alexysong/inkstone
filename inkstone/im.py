# -*- coding: utf-8 -*-

import scipy.linalg as sla


def im(phi1, psi1, phi2, psi2, phi1_is_idt=False, psi1_is_idt=False):
    """
    calculate the interface matrix from layer 1 to layer 2
    This works for 2D TE, TM or 3D.

    Parameters
    ----------
    phi1        :   ndarray
    psi1        :   ndarray
    phi2        :   ndarray
    psi2        :   ndarray
    phi1_is_idt :   bool
                    if phi1 is identity matrix
    psi1_is_idt :   bool
                    if psi1 is identity matrix

    Returns
    -------
    a12     :   ndarray
    b12     :   ndarray

    """

    if phi1_is_idt:
        term1 = phi2
    else:
        term1 = sla.solve(phi1, phi2)
    if psi1_is_idt:
        term2 = psi2
    else:
        term2 = sla.solve(psi1, psi2)

    a12 = term1 + term2
    b12 = term1 - term2

    return a12, b12
