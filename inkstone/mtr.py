# -*- coding: utf-8 -*-

from typing import Tuple, Union
import numpy as np
# import numpy.linalg as la
# from warnings import warn


class Mtr:
    # material need to have a name such that user can access it by its name
    def __init__(self,
                 epsi: Union[Union[float, complex], Tuple[Union[float, complex], Union[float, complex], Union[float, complex]], np.ndarray],
                 mu: Union[Union[float, complex], Tuple[Union[float, complex], Union[float, complex], Union[float, complex]], np.ndarray],
                 name=None):
        """
        Material.
        """
        self._epsi = None
        self._mu = None
        self._epsi_inv = None
        self._mu_inv = None

        self.epsi = epsi
        self.mu = mu

        self.name = name

    @property
    def epsi(self) -> np.ndarray:
        return self._epsi

    @epsi.setter
    def epsi(self, val):
        if (type(val) is float) or (type(val) is int) or (type(val) is complex) or (type(val) is np.float64) or (type(val) is np.complex128):
            val = np.eye(3, dtype=complex) * val
        elif np.array(val).ndim == 1 and np.array(val).size == 3:
            val = np.diag(val) + 0j
        elif (type(val) is not np.ndarray) or val.shape != (3, 3):
            raise ValueError('data format for epsilon is incorrect.')
        self._epsi = val + 0j
        # explicit computation of inverse.
        v = val
        adbc = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
        if adbc == 0 or v[2, 2] == 0:
            raise Exception('Singular permittivity tensor.')
        ei = np.array([[v[1, 1] / adbc, -v[0, 1] / adbc, 0],
                       [-v[1, 0] / adbc, v[0, 0] / adbc, 0],
                       [0, 0, 1 / v[2, 2]]], dtype=complex)
        self._epsi_inv = ei

    @property
    def mu(self) -> np.ndarray:
        return self._mu

    @mu.setter
    def mu(self, val):
        if (type(val) is float) or (type(val) is int) or (type(val) is complex) or (type(val) is np.float64) or (type(val) is np.complex128):
            val = np.eye(3, dtype=complex) * val
        elif np.array(val).ndim == 1 and np.array(val).size == 3:
            val = np.diag(val) + 0j
        elif (type(val) is not np.ndarray) or val.shape != (3, 3):
            raise ValueError('data format for mu is incorrect.')
        self._mu = val + 0j
        # explicit computation of inverse.
        v = val
        adbc = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
        if adbc == 0 or v[2, 2] == 0:
            raise Exception('Singular permittivity tensor.')
        mi = np.array([[v[1, 1] / adbc, -v[0, 1] / adbc, 0],
                       [-v[1, 0] / adbc, v[0, 0] / adbc, 0],
                       [0, 0, 1 / v[2, 2]]], dtype=complex)
        self._mu_inv = mi

    @property
    def epsi_inv(self) -> np.ndarray:
        return self._epsi_inv

    @property
    def mu_inv(self) -> np.ndarray:
        return self._mu_inv

