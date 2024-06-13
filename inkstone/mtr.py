# -*- coding: utf-8 -*-

from typing import Tuple, Union, Optional
from GenericBackend import genericBackend as gb
# import numpy.linalg as la
# from warnings import warn


class Mtr:
    # material need to have a name such that user can access it by its name
    def __init__(self,
                 epsi: Union[Union[float, complex], Tuple[Union[float, complex], Union[float, complex], Union[float, complex]], any],
                 mu: Union[Union[float, complex], Tuple[Union[float, complex], Union[float, complex], Union[float, complex]], any],
                 name=None,
                 gb=gb):
        """
        Material.
        """
        self.gb = gb
        self._epsi: Optional[any] = None
        self._mu: Optional[any] = None
        self._epsi_inv: Optional[any] = None
        self._mu_inv: Optional[any] = None

        self._ep_is_vac = True
        self._mu_is_vac = True
        self.is_vac: bool = True

        self.is_diagonal: bool = True
        self.is_isotropic: bool = True
        self.is_dege: bool = True

        self.ep_is_diagonal: bool = True
        self.ep_is_isotropic: bool = True

        self.mu_is_diagonal: bool = True
        self.mu_is_isotropic: bool = True

        self.epsi = epsi
        self.mu = mu

        self.name = name
        

    @property
    def ep_is_diagonal(self) -> bool:
        return self._ep_is_diagonal

    @ep_is_diagonal.setter
    def ep_is_diagonal(self, val):
        self._ep_is_diagonal = val
        self._check_diagonal()
        self._check_isotropic()
        self._check_degenerate()

    @property
    def ep_is_isotropic(self) -> bool:
        return self._ep_is_isotropic

    @ep_is_isotropic.setter
    def ep_is_isotropic(self, val):
        self._ep_is_isotropic = val
        self._check_diagonal()
        self._check_isotropic()
        self._check_degenerate()

    @property
    def ep_is_vac(self) -> bool:
        return self._ep_is_vac

    @ep_is_vac.setter
    def ep_is_vac(self, val):
        self._ep_is_vac = val
        self._check_vac()

    @property
    def mu_is_diagonal(self) -> bool:
        return self._mu_is_diagonal

    @mu_is_diagonal.setter
    def mu_is_diagonal(self, val):
        self._mu_is_diagonal = val
        self._check_diagonal()
        self._check_isotropic()
        self._check_degenerate()

    @property
    def mu_is_isotropic(self) -> bool:
        return self._mu_is_isotropic

    @mu_is_isotropic.setter
    def mu_is_isotropic(self, val):
        self._mu_is_isotropic = val
        self._check_diagonal()
        self._check_isotropic()
        self._check_degenerate()

    @property
    def mu_is_vac(self) -> bool:
        return self._mu_is_vac

    @mu_is_vac.setter
    def mu_is_vac(self, val):
        self._mu_is_vac = val
        self._check_vac()

    @property
    def epsi(self) -> any:
        return self._epsi

    @epsi.setter
    def epsi(self, val):
        if type(val) in [float, int, complex, self.gb.float64, self.gb.complex128]:
            ep = self.gb.eye(3, dtype=self.gb.complex128) * val
            self._epsi = ep + 0j
            self.ep_is_diagonal = True
            self.ep_is_isotropic = True
            if val == 1.:
                self.ep_is_vac = True
            else:
                self.ep_is_vac = False
        elif self.gb.parseData(val).ndim == 1 and self.gb.getSize(self.gb.parseData(val)) == 3:
            ep = self.gb.diag(val) + 0j
            self._epsi = ep + 0j
            self.ep_is_diagonal = True
            if check_iso(ep):
                self.ep_is_isotropic = True
            else:
                self.ep_is_isotropic = False
            if val[0] == 1. and val[1] == 1. and val[2] == 1.:
                self.ep_is_vac = True
            else:
                self.ep_is_vac = False
        elif (type(val) is not any) or val.shape != (3, 3):
            raise ValueError('data format for epsilon is incorrect.')
        else:
            ep = val
            self._epsi = val + 0j
            if check_diag(val):
                self.ep_is_diagonal = True
                if check_iso(val):
                    self.ep_is_isotropic = True
                    if val[0, 0] == 1. and val[1, 1] == 1. and val[2, 2] == 1.:
                        self.ep_is_vac = True
                    else:
                        self.ep_is_vac = False
                else:
                    self.ep_is_isotropic = False
                    self.ep_is_vac = False
            else:
                self.ep_is_diagonal = False
                self.ep_is_isotropic = False
                self.ep_is_vac = False

        # explicit computation of inverse.
        v = ep
        adbc = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
        if adbc == 0 or v[2, 2] == 0:
            raise Exception('Singular permittivity tensor.')
        ei = self.gb.parseData([[v[1, 1] / adbc, -v[0, 1] / adbc, 0],
                       [-v[1, 0] / adbc, v[0, 0] / adbc, 0],
                       [0, 0, 1 / v[2, 2]]], dtype=self.gb.complex128)
        self._epsi_inv = ei

    @property
    def mu(self) -> any:
        return self._mu

    @mu.setter
    def mu(self, val):
        if type(val) in [float, int, complex, self.gb.float64, self.gb.complex128]:
            mu = self.gb.eye(3, dtype=self.gb.complex128) * val
            self._mu = mu + 0j
            self.mu_is_diagonal = True
            self.mu_is_isotropic = True
            if val == 1.:
                self.mu_is_vac = True
            else:
                self.mu_is_vac = False
        elif self.gb.parseData(val).ndim == 1 and self.gb.getSize(self.gb.parseData(val)) == 3:
            mu = self.gb.diag(val) + 0j
            self._mu = mu + 0j
            self.mu_is_diagonal = True
            if check_iso(mu):
                self.mu_is_isotropic = True
            else:
                self.mu_is_isotropic = False
            if val[0] == 1. and val[1] == 1. and val[2] == 1.:
                self.mu_is_vac = True
            else:
                self.mu_is_vac = False
        elif (type(val) is not any) or val.shape != (3, 3):
            raise ValueError('data format for mu is incorrect.')
        else:
            mu = val
            self._mu = val + 0j
            if check_diag(val):
                self.mu_is_diagonal = True
                if check_iso(val):
                    self.mu_is_isotropic = True
                    if val[0, 0] == 1. and val[1, 1] == 1. and val[2, 2] == 1.:
                        self.mu_is_vac = True
                    else:
                        self.mu_is_vac = False
                else:
                    self.mu_is_isotropic = False
                    self.mu_is_vac = False
            else:
                self.mu_is_diagonal = False
                self.mu_is_isotropic = False
                self.mu_is_vac = False

        # explicit computation of inverse.
        v = mu
        adbc = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
        if adbc == 0 or v[2, 2] == 0:
            raise Exception('Singular permittivity tensor.')
        mi = self.gb.parseData([[v[1, 1] / adbc, -v[0, 1] / adbc, 0],
                       [-v[1, 0] / adbc, v[0, 0] / adbc, 0],
                       [0, 0, 1 / v[2, 2]]], dtype=self.gb.complex128)
        self._mu_inv = mi

    @property
    def epsi_inv(self) -> any:
        return self._epsi_inv

    @property
    def mu_inv(self) -> any:
        return self._mu_inv

    def _check_isotropic(self):
        """check if material is isotropic."""
        if (self.epsi is not None) and (self.mu is not None):
            if self.ep_is_isotropic and self.mu_is_isotropic:
                self.is_isotropic = True
            else:
                self.is_isotropic = False

            # ep = self.epsi
            # mu = self.mu
            # if check_iso(ep) and check_iso(mu):
            #     self.is_isotropic = True
            # else:
            #     self.is_isotropic = False

    def _check_diagonal(self):
        """check if material is diagonal."""
        if (self.epsi is not None) and (self.mu is not None):
            if self.ep_is_diagonal and self.mu_is_diagonal:
                self.is_diagonal = True
            else:
                self.is_diagonal = False

            # ep = self.epsi
            # mu = self.mu
            # if check_diag(ep) and check_diag(mu):
            #     self.is_diagonal = True
            # else:
            #     self.is_isotropic = False

    def _check_degenerate(self):
        """check if material is degenerate eigen"""
        if (self.epsi is not None) and (self.mu is not None):
            if self.ep_is_diagonal and self.mu_is_diagonal:
                if self.ep_is_isotropic and self.mu_is_isotropic:
                    self.is_dege = True
                else:
                    ep = self.epsi
                    mu = self.mu
                    if check_dege(ep, mu):
                        self.is_dege = True
                    else:
                        self.is_dege = False
            else:
                self.is_dege = False

    def _check_vac(self):
        if self.ep_is_vac and self.mu_is_vac:
            self.is_vac = True
        else:
            self.is_vac = False

def check_iso(val):
    """check of 3x3 tensor is isotropic"""
    if check_diag(val):
        if (val[0, 0] == val[1, 1]) and (val[0, 0] == val[2, 2]):
            return True
        else:
            return False
    else:
        return False


def check_diag(val):
    """check of 3x3 tensor is diagonal"""
    if (val[0, 1] == 0.) and (val[0, 2] == 0.) and (val[1, 0] == 0.) and (val[1, 2] == 0.) and (val[2, 0] == 0.) and (val[2, 1] == 0.):
        return True
    else:
        return False


def check_dege(ep, mu):
    """check if epsi and mu 3x3 tensors satisfy degenerate condition"""
    if check_diag(ep) and check_diag(mu):
        mxx, mxy, mxz, myx, myy, myz, mzx, mzy, mzz = mu.ravel()
        exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz = ep.ravel()
        if (eyy * mzz == ezz * myy) and (exx * mzz == ezz * mxx):
            return True
        else:
            return False
    else:
        return False
