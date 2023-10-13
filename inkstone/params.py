# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
from scipy import sparse as sps
import scipy.linalg as sla
# import scipy.fft as fft
from typing import Tuple, List, Union, Optional, Set
# import time
from warnings import warn
from inkstone.recipro import recipro
from inkstone.g_pts import g_pts
from inkstone.g_pts_1d import g_pts_1d
from inkstone.max_idx_diff import max_idx_diff
from inkstone.conv_mtx_idx import conv_mtx_idx_2d


class Params:
    """
    Contains various parameters that are relevant to solver and shared between solver and each layer.
    """

    def __init__(self,
                 latt_vec=None,
                 num_g=None,
                 frequency=None,
                 omega=None,
                 theta=None,
                 phi=None,
                 show_calc_time = False
                 ):
        """

        Parameters
        ----------
        latt_vec        :   tuple[tuple[float, float], tuple[float, float]]
                            primitive lattice vectors
        num_g           :   int
                            number of g points
        frequency       :   float or complex
        omega           :   float or complex
        theta, phi      :   float
                            incident angles in degrees.
                            Theta is the angle between incident k and z axis.
                            phi is the angle between the in-plane projection of k and the x axis. Rotating from kx axis phi degrees ccw around z axis to arrive at the kx of incident wave.
        """

        self.gs: Optional[List[Tuple[float, float]]] = None  # list of g points for E and H fields. Added by k_inci to get the ks, i.e. k points for E and H
        self.idx_g: Optional[List[Tuple[int, int]]] = None  # list of g points indices
        self.idx_conv_mtx: Optional[np.ndarray] = None  # indexing array to for constructing convolution matrices.
        self.ks: Optional[List[Tuple[float, float]]] = None  # list of k points for E and H fields.
        self.idx_g_ep_mu: Optional[List[Tuple[int, int]]] = None
        self.idxa_g_ep_mu: Optional[Tuple[np.ndarray, np.ndarray]] = None  # The g indices for ep and mu.
        self.idx_g_ep_mu_used: Optional[List[Tuple[int, int]]] = None  # the actually used epsilon and mu grid points
        self.ks_ep_mu: Optional[List[Tuple[float, float]]] = None  # list of k points where epsi and mu Fourier components are to be calculated
        self.ka_ep_mu: Optional[Tuple[np.ndarray, np.ndarray]] = None  # The k points for ep and mu, Tuple of meshgrid (kx, ky).
        self.Kx: Optional[np.ndarray] = None  # the diagonals of the Big diagonal matrix
        self.Ky: Optional[np.ndarray] = None  # the diagonals of the Big diagonal matrix
        self.Kz: Optional[np.ndarray] = None  # the diagonals of the Big diagonal matrix, useful for uniform layers
        self.mmax: Optional[int] = None  # max index in b1 direction for epsi and mu Fourier series
        self.nmax: Optional[int] = None  # max index in b2 direction for epsi and mu Fourier series
        self.phi0: Optional[np.ndarray] = None  # E field eigen mode in vacuum (identity matrix), side length 2*num_g
        self.psi0: Optional[np.ndarray] = None  # H field eigen mode in vacuum (Q * Phi * q^-1), side length 2*num_g
        self.phi0_2x2s: Optional[np.ndarray] = None  # the non-zero elements of phi0 stored in (2, 2, num_g) shape
        self.phif: Optional[np.ndarray] = None  # E field eigen mode in fic material, shape (2num_g, 2num_g)
        self.psif: Optional[np.ndarray] = None  # H field eigen mode in fic material, shape (2num_g, 2num_g)
        self.q0: Optional[np.ndarray] = None  # 1d array, eigen propagation constant in z direction in vacuum, length 2*num_g
        self.q0_half: Optional[np.ndarray] = None  # 1d array, eigen propagation constant in z direction in vacuum, length num_g
        self.q0_0: Optional[np.ndarray] = None  # 1d array, containing idxs to the idx_g list which q0 is 0, i.e. parallel to surface
        self.q0_inv: Optional[np.ndarray] = None  # 1d array, 1./q0, elementwise inversion of q0, length 2*num_g
        self.P0_val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None  # Tuple of 4, each is an ndarray of size num_g, containing the diagonal elements of the 4 blocks of P0.
        self.Q0_val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None  # Tuple of 4, each is an ndarray of size num_g, containing the diagonal elements of the 4 blocks of Q0.
        self.im0: Optional[Tuple[np.ndarray, np.ndarray]] = None  # vacuum interface matrix
        self.sm0: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None  # vacuum scattering matrix, (s11_0, s12_0, s21_0, s22_0), each of s_ij has side length 2*num_g
        self._rad_cha_0: Optional[List[int]] = None

        self.ccnif = "physical"
        self.ccpif = "ac"

        self.__num_g_actual: Optional[int] = None  # actual number of G points used.
        self._num_g_input: Optional[int] = None  # number of G points input by user
        self._k_inci: Optional[Tuple[Tuple[float, float]], Tuple[float, float]] = None  # incident in-plane k
        self._omega: Optional[complex] = None
        self._frequency: Optional[complex] = None
        self._recipr_vec: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self._latt_vec: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.is_1d_latt: bool = False  # if this is solving a 2d structure. Incident wave can still have phi rotation, i.e. 3D scenario for a 2D structure.
        self._theta: Optional[float] = None  # in rad
        self._phi: Optional[float] = None  # in rad
        self._uc_area: Optional[float] = None
        self.cos_varthetas: Optional[List[float]] = None  # incident angles
        self.sin_varthetas: Optional[List[float, complex]] = None
        self.cos_phis: Optional[List[float]] = None
        self.sin_phis: Optional[List[float]] = None

        self._incident_orders = None
        self._incident_orders_bk = None
        self._s_amps = None
        self._p_amps = None
        self._s_amps_bk = None
        self._p_amps_bk = None
        self.ai: Optional[np.ndarray] = None
        self.bo: Optional[np.ndarray] = None

        # self.q0_contain_0: bool = False

        self.show_calc_time = show_calc_time

        # property initialization can change other attributes hence must be after initialization of other attributes.
        if latt_vec is not None:
            self.latt_vec = latt_vec
        if num_g is not None:
            self.num_g = num_g
        if frequency is not None:
            self.frequency = frequency
        if omega is not None:
            self.omega = omega
        if theta is not None:
            self.theta = theta
        if phi is not None:
            self.phi = phi

    @property
    def latt_vec(self) -> Union[float, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Primitive lattice vectors.
        The lattice vector and be a single number which means 1D, or tuples such as ((a1x, a1y), (a2x, a2y)) which means 2D .

        If tuples are supplied but one of them is (0, 0), then it is also a 1d. In this case the lattice will be rotated such that the actual 1d direction is in x axis where y is uniform.
        Note the incident wave angles will be w.r.t. the structure AFTER it's rotated to x axis.

        """
        return self._latt_vec

    @latt_vec.setter
    def latt_vec(self, val: Union[float, Tuple[Tuple[float, float], Tuple[float, float]]]):
        if type(val) is not tuple:
            val = ((val, 0), (0, 0))
            self.is_1d_latt = True
        else:
            a, b = val
            an, bn = [la.norm(a), la.norm(b)]
            # if either latt vec is zero, then it is 1D. Make sure it's x-z for layer solver.
            if an == 0:
                val = ((bn, 0), (0, 0))
                self.is_1d_latt = True
                warn("2D structure (in-plane 1D), non-uniform direction is rotated to x axis.")
            if bn == 0:
                val_old = val
                val = ((an, 0), (0, 0))
                self.is_1d_latt = True
                if val_old != val:
                    warn("2D structure (in-plane 1D), non-uniform direction is rotated to x axis.")
        self._latt_vec: Union[Tuple[Tuple[float, float], Tuple[float, float]]] = val
        self._recipr_vec: Tuple[Tuple[float, float], Tuple[float, float]] = recipro(val[0], val[1])
        self.if_2d()
        self._calc_gs()
        self._calc_uc_area()
        # self._calc_ks_ep_mu()  # called through _calc_gs -> _calc_ks

    @property
    def recipr_vec(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Reciprocal lattice vectors"""
        return self._recipr_vec

    def if_2d(self):
        b1, b2 = self._recipr_vec
        if b1[0] == float('inf') or b1[1] == float('inf') or b2[0] == float('inf') or b2[1] == float('inf'):
            self.is_1d_latt = True
        else:
            self.is_1d_latt = False

    @property
    def num_g(self) -> int:
        """The number of G points to be included. To ensure symmetry, the actual num_g may be slightly different than the user set value."""
        return self._num_g_ac

    @num_g.setter
    def num_g(self, val: int):
        self._num_g_input: int = val
        self._calc_gs()

    @property
    def _num_g_ac(self) -> int:
        """actual number of G points"""
        return self.__num_g_actual

    @_num_g_ac.setter
    def _num_g_ac(self, val: int):
        # todo: all other calc should use this than original num_g
        self.__num_g_actual = val
        # self._calc_phi0()
        self._calc_phi0_psi0()
        self._calc_phif_psif()
        # self._calc_q0()  # called through _calc_gs - _calc_ks
        self._calc_im0()
        self._calc_s_0()
        # self.calc_ai_bo_3d()  # called through _calc_gs

    @property
    def frequency(self) -> Union[float, complex]:
        """ frequency, c/lambda in vacuum """
        return self._frequency

    @frequency.setter
    def frequency(self, val: Union[float, complex]):
        if val is not None:
            self.omega = val * np.pi * 2.
            self._frequency = val

    @property
    def omega(self) -> Union[float, complex]:
        """ angular frequency, 2pi c/lambda in vacuum """
        return self._omega

    @omega.setter
    def omega(self, val: Union[float, complex]):
        if val is not None:
            self._omega = val
            self._frequency = val / np.pi / 2.
            # self.q0_contain_0 = False
            self._calc_gs()  # recalculating gs because some g may be possibly removed in previous runs to remove Wood's anomaly.
            self._calc_k_inci()
            # self._calc_q0()  # called through _calc_k_inci - _calc_ks
            # self._calc_P0Q0()  # called through _calc_k_inci - _calc_ks - _calc_Km
            # self._calc_angles()  # called through _calc_k_inci - _calc_ks -

    @property
    def k_inci(self) -> Tuple[float, float]:
        """kx and ky projection of the incident wave vector"""
        return self._k_inci

    @property
    def theta(self) -> Union[float, complex]:
        """he angle between incident k and z axis, in degrees, range: [0, pi/2]"""
        if self._theta is not None:
            theta = self._theta / np.pi * 180.
        else:
            theta = None
        return theta

    @theta.setter
    def theta(self, val: Union[float, complex]):
        if val is not None:
            self._theta: float = val * np.pi / 180.
            self._calc_k_inci()
            self._calc_angles()
        else:
            if self._theta is None:
                self._theta = 0.

    @property
    def phi(self) -> float:
        """ the angle between the in-plane projection of k and the x axis in degrees. Rotating from kx axis phi degrees ccw around z axis to arrive at the kx of incident wave. """
        if self._phi is not None:
            phi = self._phi / np.pi * 180.
        else:
            phi = None
        return phi

    @phi.setter
    def phi(self, val: float):
        if val is not None:
            self._phi: float = val * np.pi / 180.
            self._calc_k_inci()
            self._calc_angles()
        else:
            if self._phi is None:
                self._phi = 0.

    @property
    def rad_cha_0(self) -> List[int]:
        """list of indices in `self.idx_g` that corresponds to radiation channels in vacuum"""
        return self._rad_cha_0  # between 0 and num_g

    @property
    def ccnif(self) -> str:
        """
        {"physical", "ac"}, default: "physical".
        """
        return self._ccnif

    @ccnif.setter
    def ccnif(self, val):
        if (val == "physical") or (val == "ac"):
            self._ccnif = val
        else:
            warn("ccnif not understood and not changed in the solver.")

    @property
    def ccpif(self) -> str:
        """
        {"physical", "ac"}, default: "ac".
        """
        return self._ccpif

    @ccpif.setter
    def ccpif(self, val):
        if (val == "physical") or (val == "ac"):
            self._ccpif = val
        else:
            warn("ccpif not understood and not changed in the solver.")

    def set_inci_ord_amp(self,
                         s_amplitude: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                         p_amplitude: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                         order: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
                         s_amplitude_back: Union[float, List[float]] = None,
                         p_amplitude_back: Union[float, List[float]] = None,
                         order_back: Union[Tuple[int, int], List[Tuple[int, int]]] = None
                         ):
        """
        Setting and resetting incident orders and amplitudes.

        If any of the optional arguments are not given, then use the previously set values.

        Parameters
        ----------
        s_amplitude
        p_amplitude
        order
        s_amplitude_back
        p_amplitude_back
        order_back
        """
        # Convert order and order_back to lists (can be empty list)
        # todo: how do I know the user chosen order is in idx_g?
        if order is None:
            if self._incident_orders is None:
                self._incident_orders = [(0, 0)]
        elif type(order) is tuple:
            order = [order]
            self._incident_orders = order
        else:
            self._incident_orders = order

        if order_back is None:
            if self._incident_orders_bk is None:
                self._incident_orders_bk = [(0, 0)]
        elif type(order_back) is tuple:
            order_back = [order_back]
            self._incident_orders_bk = order_back
        else:
            self._incident_orders_bk = order_back

        # convert s and p amplitudes to lists (can be empty list)
        amp = []
        for amps, selfamps, od in zip([[s_amplitude, p_amplitude], [s_amplitude_back, p_amplitude_back]], [[self._s_amps, self._p_amps], [self._s_amps_bk, self._p_amps_bk]], [self._incident_orders, self._incident_orders_bk]):
            for a, s in zip(amps, selfamps):
                if a is None:
                    if s is None:
                        a = [0. for i in range(len(od))]
                    else:
                        a = s
                elif not hasattr(a, "__len__"):
                    a = [a]
                amp.append(a)
        self._s_amps = amp[0]
        self._p_amps = amp[1]
        self._s_amps_bk = amp[2]
        self._p_amps_bk = amp[3]

        if (len(self._incident_orders) != len(self._s_amps)) or (len(self._incident_orders) != len(self._p_amps)):
            raise Exception("The list length of the incident s amplitudes, p amplitudes, and incident orders are not equal.")

        if (len(self._incident_orders_bk) != len(self._s_amps_bk)) or (len(self._incident_orders_bk) != len(self._p_amps_bk)):
            raise Exception("The list length of the backside incident s amplitudes, p amplitudes, and incident orders are not equal.")

        self.calc_ai_bo_3d()

    def _calc_k_inci(self):
        if (self.theta is not None) and (self.phi is not None) and (self.omega is not None):
            kx = self.omega.real * np.cos(np.pi/2 - self._theta) * np.cos(self._phi)
            ky = self.omega.real * np.cos(np.pi/2 - self._theta) * np.sin(self._phi)
            # todo: using self.omega (kx ky complex) also gives answers, physical meaning is different
            self._k_inci: Tuple[float, float] = (kx, ky)
            self._calc_ks()

    def _calc_gs(self):
        """ calculate E and H Fourier components g points """
        if self._num_g_input and self.recipr_vec:
            b1, b2 = self.recipr_vec
            b1n, b2n = [la.norm(b) for b in [b1, b2]]
            if b1n != float('inf') and b2n != float('inf'):
                self.gs, self.idx_g = g_pts(self._num_g_input, self.recipr_vec[0], self.recipr_vec[1])
            elif b1n == float('inf'):
                self.gs, idx = g_pts_1d(self._num_g_input, b2)
                self.idx_g = [(0, i) for i in idx]
            elif b2n == float('inf'):
                self.gs, idx = g_pts_1d(self._num_g_input, b1)
                self.idx_g = [(i, 0) for i in idx]
            else:
                raise Exception("Both reciprocal lattice vectors are infinite. Can't calculate g points.")
            self._num_g_ac = len(self.gs)
            self._calc_ks()
            self._calc_conv_mtx_idx()
            # self.calc_ai_bo_3d()  # called through _calc_ks() - _calc_angles()

    def _remove_gs(self,
                   idxs_rm: List[Tuple[int, int]]):
        """
        remove certain g points by given list of g indices such as [(1, -2)]

        Parameters
        ----------
        idxs_rm:
            indices to remove, each element in this list should be a tuple of integers like (1, -2)

        Returns
        -------

        """
        idx_g_new = []
        gs_new = []
        for ii, idx in enumerate(self.idx_g):
            to_add = True
            for idx_rm in idxs_rm:
                if idx == idx_rm:
                    to_add = False
                    break
            if to_add:
                idx_g_new.append(idx)
                gs_new.append(self.gs[ii])
        self.idx_g = idx_g_new
        self.gs = gs_new
        self._num_g_ac = len(self.gs)
        self._calc_ks()
        self._calc_conv_mtx_idx()

    def _remove_gs_xuhao(self,
                         idxs: List[int]):
        """
        Remove g points by the index in the current idx_g list, for example, idxs=[3, 5] means to remove the 3rd and the 5th element in the g list.

        Parameters
        ----------
        idxs

        Returns
        -------

        """
        for ii in sorted(idxs, reverse=True):
            del self.gs[ii]
            del self.idx_g[ii]
        self._num_g_ac = len(self.gs)
        self._calc_ks()
        self._calc_conv_mtx_idx()

    def _calc_ks(self):
        if self.gs and self.k_inci:
            self.ks = [(g[0]+self.k_inci[0], g[1] + self.k_inci[1]) for g in self.gs]
            self._calc_q0()
            self._calc_Km()
            self._calc_ks_ep_mu()
            self._calc_angles()

    def _calc_Km(self):
        """Calculate Kx Ky arrays"""
        if self.ks:
            # t1 = time.process_time()
            ksa = np.array(self.ks)  # shape (NumG, 2)
            kx = ksa[:, 0]
            ky = ksa[:, 1]
            self.Kx = kx
            self.Ky = ky

            # print('_calc_Km', time.process_time() - t1)

            self._calc_P0Q0()
            # self._calc_phi0_psi0()  # called through _calc_P0Q0()

    def _calc_conv_mtx_idx(self):
        if self.idx_g:
            self.idx_conv_mtx = conv_mtx_idx_2d(self.idx_g, self.idx_g)
            self.idx_g_ep_mu_used = list(set([(i, j) for (i, j) in self.idx_conv_mtx.reshape(self._num_g_ac ** 2, 2)]))  # The first element of each tuple is in physical "x" direction

    def _calc_ks_ep_mu(self):
        if self.idx_g:
            # t1 = time.process_time()

            m, n = max_idx_diff(self.idx_g)
            self.mmax = m
            self.nmax = n
            x = np.arange(-m, m+1)
            y = np.arange(-n, n+1)
            xx, yy = np.meshgrid(x, y)

            # t2 = time.process_time()

            self.idx_g_ep_mu = list(zip(xx.ravel(), yy.ravel()))
            self.idxa_g_ep_mu = (xx, yy)

            # t3 = time.process_time()

            b1, b2 = self.recipr_vec
            b1n, b2n = [la.norm(b) for b in [b1, b2]]
            # if one of them is infinity, this is a 1D structure. set it to 0 such that they don't appear in ks_ep_mu
            if b1n == float('inf'):
                b1 = (0, 0)
            if b2n == float('inf'):
                b2 = (0, 0)

            kx = xx * b1[0] + yy * b2[0]
            ky = xx * b1[1] + yy * b2[1]
            self.ks_ep_mu = list(zip(kx.ravel(), ky.ravel()))  # each element is (kx, ky) tuple.
            self.ka_ep_mu = (kx, ky)

            # print(t2 - t1)
            # print(t3 - t2)
            # print(time.process_time() - t3)
            # print('_calc_ks_ep_mu', time.process_time()-t1)

    def _calc_q0(self):
        if self._num_g_ac and (self.omega is not None) and self.ks:
            # t1 = time.process_time()
            k_parallel = np.linalg.norm(np.array(self.ks), axis=-1)
            q02 = np.ones(self._num_g_ac) * np.square(self.omega) - np.square(k_parallel) + 0j
            self._rad_cha_0 = np.where(q02.real > 0)[0].tolist()  # todo: even for radiation channel, if omega.imag larger than omega.real, q02.real is negative
            q0 = np.sqrt(q02)
            if self.omega.imag < 0:
                if self.ccnif == "physical":
                    q0[(q02.real < 0) * (q0.imag < 0)] *= -1
                elif self.ccnif == "ac":
                    q0[(q02.real < 0) * (q0.imag > 0)] *= -1
                else:
                    warn("ccnif not recognized. Default to 'physical'.")
                    q0[(q02.real < 0) * (q0.imag < 0)] *= -1
            elif self.omega.imag > 0:
                if self.ccpif == "ac":
                    q0[(q02.real < 0) * (q0.imag < 0)] *= -1
                elif self.ccpif == "physical":
                    q0[(q02.real < 0) * (q0.imag > 0)] *= -1
                else:
                    warn("ccpif not recognized. Default to 'ac'.")
                    q0[(q02.real < 0) * (q0.imag < 0)] *= -1
            else:
                q0[q0.imag < 0] *= -1
            # todo: what to do at q02.real == 0 case (Woods)?

            self.q0_0 = np.where(np.abs(q0) == 0.)[0]

            self.q0 = np.concatenate([q0, q0])
            self.q0_half = q0
            # print('_calc_q0', time.process_time() - t1)

            with np.errstate(divide='ignore', invalid='ignore'):
                self.q0_inv = 1. / self.q0
                ii = np.where(self.q0 == 0.)
                self.q0_inv[ii] = float('inf')

            # self._calc_psi0()
            self._calc_phi0_psi0()

    def _calc_P0Q0(self):
        if (self.omega is not None) and (self.Kx is not None) and (self.Ky is not None):
            # t1 = time.process_time()

            Kx = self.Kx
            Ky = self.Ky
            o = self.omega

            P11 = 1. / o * Kx * Ky
            P12 = o - 1. / o * Kx * Kx
            P21 = -o + 1. / o * Ky * Ky
            P22 = - 1. / o * Ky * Kx

            # t2 = time.process_time()
            # print('P0 Q0 blocks', t2 - t1)

            self.P0_val = (P11, P12, P21, P22)
            self.Q0_val = self.P0_val

            # # for debugging
            # ng = self._num_g_ac
            # self.P0 = np.zeros((2*ng, 2*ng), dtype=complex)
            # r1 = range(ng)
            # r2 = range(ng, 2 * ng)
            # self.P0[r1, r1] = P11
            # self.P0[r2, r1] = P21
            # self.P0[r1, r2] = P12
            # self.P0[r2, r2] = P22
            # self.P0 = np.block([[np.diag(P11), np.diag(P12)],
            #                     [np.diag(P21), np.diag(P22)]])

            # self._calc_psi0()
            self._calc_phi0_psi0()
            self._calc_phif_psif()

    def _calc_phi0_psi0(self):
        if (self.Q0_val is not None) and (self.q0 is not None) and (self._num_g_ac is not None):
            # t1 = time.process_time()

            o = self.omega
            Kx = self.Kx.copy()
            Ky = self.Ky.copy()
            k_norm = np.sqrt(np.conj(Kx) * Kx + np.conj(Ky) * Ky)

            # # Use [ky, -kx] etc. as eigen
            # Tx = Kx  # term with x subscript
            # Ty = Ky
            #
            # # for small k just use [1, 0], [0, 1] as eigen.
            # skc = 0.5  # small k criterion
            # small_k = np.where(k_norm < (skc * np.abs(o)))[0]
            # Tx[small_k] = 0.
            # Ty[small_k] = 1.
            # t_norm = k_norm.copy()  # term norm
            # t_norm[small_k] = 1.
            #
            # Txm = np.diag(Tx)
            # Tym = np.diag(Ty)
            # jwTxq = 1j / o * Tx * self.q0_half
            # jwTyq = 1j / o * Ty * self.q0_half
            #
            # # for small k, need separate calc here
            # jwTxq[small_k] = -1j / o * Kx[small_k] * Ky[small_k] / self.q0_half[small_k]
            # jwTyq[small_k] = -1j / o * (-np.conj(Kx[small_k])*Kx[small_k] - np.conj(self.q0_half[small_k])*self.q0_half[small_k]) / self.q0_half[small_k]
            #
            # jwTxqm = np.diag(jwTxq)
            # jwTyqm = np.diag(jwTyq)
            #
            # ng = self._num_g_ac
            # r1 = range(ng)
            # r2 = range(ng, 2 * ng)
            # phi0 = np.zeros((2*ng, 2*ng), dtype=complex)
            # psi0 = phi0.copy()
            # phi0[r1, r1] = Ty
            # phi0[r2, r1] = -Tx
            # phi0[r1, r2] = jwTxq
            # phi0[r2, r2] = jwTyq
            # psi0[r1, r1] = jwTxq
            # psi0[r2, r1] = jwTyq
            # psi0[r1, r2] = Ty
            # psi0[r2, r2] = -Tx
            #
            # # phi0 = np.block([[Tym, jwTxqm],
            # #                  [-Txm, jwTyqm]])
            # # psi0 = np.block([[jwTxqm, Tym],
            # #                  [jwTyqm, -Txm]])
            #
            # self.phi0_2x2s = np.array([[Ty, jwTxq],
            #                           [-Tx, jwTyq]])


            # with normalization
            q0h = self.q0_half
            ng = self._num_g_ac

            skc = 0.05  # small k criterion
            i_knz = np.where(k_norm < (skc * np.abs(o)))[0]  # k near zero
            i_qsw = np.where((np.abs(q0h) <= np.abs(o)) * (k_norm >= (skc * np.abs(o))))[0]  # q smaller than omega, but k_norm not near zero
            i_qlw = np.where(np.abs(q0h) >  np.abs(o))[0]  # q larger than omega

            c1 = np.array([Ky, -Kx], dtype=complex)
            c2 = np.array([Kx, Ky], dtype=complex)
            c1f = np.ones(ng, dtype=complex)  # multiplication factor which includes normalization
            c2f = c1f.copy()

            c1[:, i_knz] = np.array([[1.], [0.]])
            c2[:, i_knz] = -1j / o * np.array([Kx[i_knz] * Ky[i_knz] / q0h[i_knz], (-np.square(Kx[i_knz]) - np.square(q0h[i_knz])) / q0h[i_knz]])  # should not be |Kx|^2

            c1f[i_qlw] = o / q0h[i_qlw] / k_norm[i_qlw]
            c2f[i_qlw] = 1j / k_norm[i_qlw]

            c1f[i_qsw] = 1. / k_norm[i_qsw]
            c2f[i_qsw] = 1j / o * q0h[i_qsw] / k_norm[i_qsw]

            c1f[i_knz] = 1.
            c2f[i_knz] = 1.

            c1 *= c1f
            c2 *= c2f

            r1 = range(ng)
            r2 = range(ng, 2 * ng)
            phi0 = np.zeros((2*ng, 2*ng), dtype=complex)
            psi0 = phi0.copy()
            phi0[r1, r1] = c1[0, :]
            phi0[r2, r1] = c1[1, :]
            phi0[r1, r2] = c2[0, :]
            phi0[r2, r2] = c2[1, :]
            psi0[r1, r1] = c2[0, :]
            psi0[r2, r1] = c2[1, :]
            psi0[r1, r2] = c1[0, :]
            psi0[r2, r2] = c1[1, :]

            self.phi0_2x2s = np.array([c1, c2])

            # # debugging,  check if phi is eigen and consistent with psi
            # psi00 = -1j * self.P0 @ phi0 / self.q0
            # diff = np.abs(psi00 - psi0).max()
            # print('psi0 diff {:g}'.format(diff))
            #
            # check_eigen = self.P0 @ self.P0 @ phi0
            # diff1 = np.abs(check_eigen + phi0 * self.q0 * self.q0).max()
            # print('check eigen {:g}'.format(diff1))

            # # original
            # phi0 = np.eye(2 * self._num_g_ac, 2 * self._num_g_ac, dtype=complex)
            # ng = self._num_g_ac
            # psi0 = np.zeros((2 * ng, 2 * ng), dtype=complex)
            # r1 = range(ng)
            # r2 = range(ng, 2 * ng)
            # q0_inv = self.q0_inv
            # if self.Q0_val[0].size == self.q0_inv[:ng].size:
            #     psi0[r1, r1] = -1j * self.Q0_val[0] * q0_inv[:ng]
            #     psi0[r1, r2] = -1j * self.Q0_val[1] * q0_inv[ng:]
            #     psi0[r2, r1] = -1j * self.Q0_val[2] * q0_inv[:ng]
            #     psi0[r2, r2] = -1j * self.Q0_val[3] * q0_inv[ng:]

            self.phi0 = phi0
            self.psi0 = psi0

    def _calc_phif_psif(self):
        if self._num_g_ac:
            ng = self._num_g_ac
            phif = np.eye(2 * ng, 2 * ng, dtype=complex)
            psif = np.zeros((2 * ng, 2 * ng), dtype=complex)
            r1 = range(ng)
            r2 = range(ng, 2 * ng)
            # phif[r2, r2] = -1.  # attention! this was for debugging, but note since phif=eye, it is omitted in certain calculations
            psif[r2, r1] = 1.j
            psif[r1, r2] = -1.j
            self.phif = phif
            self.psif = psif

            # # for debugging
            # phif2 = np.eye(2 * ng, 2 * ng, dtype=complex)
            # psif2 = np.zeros((2 * ng, 2 * ng), dtype=complex)
            # phif2[r2, r2] = -1.  # attention! this was for debugging, but note since phif=eye, it is omitted in certain calculations
            # psif2[r2, r1] = 1.
            # psif2[r1, r2] = -1.
            # self.phif2 = phif2
            # self.psif2 = psif2

            # phif3 = np.zeros((2 * ng, 2 * ng), dtype=complex)
            # psif3 = np.zeros((2 * ng, 2 * ng), dtype=complex)
            # phif3[r1, r1] = 1.
            # phif3[r2, r1] = 1j
            # phif3[r1, r2] = -1j
            # phif3[r2, r2] = 1.
            # psif3 = phif3
            #
            # self.phif3 = phif3
            # self.psif3 = psif3

            # if self.P0_val is not None:
            #     phif4 = np.random.rand(2*ng, 2*ng)
            #     psif4 = -1j * self.P0 @ phif4 * self.q0_inv
            #     self.phif4 = phif4
            #     self.psif4 = psif4

            # # end of debugging

    def _calc_phi0(self):
        warn('This method is deprecated. Use `_calc_phi0_psi0` instead.', category=DeprecationWarning)
        if self._num_g_ac:
            self.phi0 = np.eye(2 * self._num_g_ac, 2 * self._num_g_ac, dtype=complex)

    def _calc_psi0(self):
        warn('This method is deprecated. Use `_calc_phi0_psi0` instead.', category=DeprecationWarning)

        if (self.Q0_val is not None) and (self.q0 is not None) and (self._num_g_ac is not None) and ((self.Kx is not None) and (self.Ky is not None)):
            # t1 = time.process_time()

            # if not self.q0_contain_0:
            ng = self._num_g_ac
            psi0 = np.zeros((2 * ng, 2 * ng), dtype=complex)
            r1 = range(ng)
            r2 = range(ng, 2 * ng)
            q0_inv = self.q0_inv
            if self.Q0_val[0].size == self.q0_inv[:ng].size:
                psi0[r1, r1] = -1j * self.Q0_val[0] * q0_inv[:ng]
                psi0[r1, r2] = -1j * self.Q0_val[1] * q0_inv[ng:]
                psi0[r2, r1] = -1j * self.Q0_val[2] * q0_inv[:ng]
                psi0[r2, r2] = -1j * self.Q0_val[3] * q0_inv[ng:]

                # # at Wood P is singular so this solving doesn't work.
                # _P = [np.diag(Qv) for Qv in self.Q0_val]
                # P = np.block([[_P[0], _P[1]],
                #               [_P[2], _P[3]]])
                # psi0 = 1j * la.solve(P, self.phi0) @ np.diag(self.q0)

            o = self.omega
            cn = np.where(np.abs(self.q0) == 0.)[0]
            cn1 = np.where(np.abs(self.q0) < 1e-2)[0]  # for debugging

            # # test setting zeros an ones
            # for ii in cn:
            #     if ii < ng:
            #         if self.Ky[ii] == 0.:
            #             psi0[ii, ii] = 0.
            #             psi0[ii+ng, ii] = 1.
            #             self.phi0[ii, ii] = 0.
            #         elif self.Kx[ii] != 0.:
            #             # psi0[ii, ii] = -self.P0_val[0][ii]
            #             # psi0[ii+ng, ii] = -self.P0_val[2][ii]
            #             # self.phi0[ii, ii] = 0.
            #             if (ii == 5) or (ii == 6):
            #                 psi0[ii, ii] = 0.
            #                 psi0[ii+ng, ii] = 0.
            #                 self.phi0[ii, ii] = 1.
            #                 self.phi0[ii+ng, ii] = -1.
            #
            #         else:
            #             psi0[ii, ii] = 1. * np.sign(self.Ky[ii])
            #             psi0[ii+ng, ii] = 0.
            #
            #         # if (self.Ky[ii] == 0) or (self.Kx[ii] == 0):
            #         #     # psi0[ii, ii] = -1j * 1. / o * self.Kx[cn]
            #         #     psi0[ii, ii] = 0.
            #         #     psi0[ii+ng, ii] = 1.
            #         # else:
            #         #     psi0[ii, ii] = self.P0_val[0][ii]
            #         #     psi0[ii+ng, ii] = self.P0_val[2][ii]
            #     else:
            #         iim = ii-ng
            #         if self.Kx[iim] == 0.:
            #             psi0[ii, ii] = 0.
            #             psi0[ii-ng, ii] = -1.
            #             self.phi0[ii, ii] = 0.
            #         elif self.Ky[iim] != 0.:
            #             # psi0[ii-ng, ii] = -self.P0_val[2][iim]
            #             # psi0[ii, ii] = -self.P0_val[3][iim]
            #             # self.phi0[ii, ii] = 0.
            #             if (ii == 14) or (ii == 15):
            #                 psi0[ii, ii] = 1.
            #                 psi0[ii-ng, ii] = -1.
            #                 self.phi0[ii, ii] = 0.
            #                 self.phi0[ii-ng, ii] = -0.
            #
            #         else:
            #             psi0[ii, ii] = 1. * np.sign(self.Kx[iim])
            #             psi0[ii-ng, ii] = 0.
            #
            #         # if (self.Ky[iim] == 0) or (self.Kx[iim] == 0):
            #         #     psi0[ii, ii] = 0.
            #         #     psi0[ii-ng, ii] = 1.
            #         # else:
            #         #     psi0[ii-ng, ii] = self.P0_val[1][iim]
            #         #     psi0[ii, ii] = self.P0_val[3][iim]
            #
            #     # self.phi0[ii, ii] = 0.

            # # this part is to test scaling of Phi and Psi
            # factor = 1e-7
            # psi0[5, 5] *= factor
            # psi0[14, 5] *= factor
            # self.phi0[5, 5] *= factor
            # for ii in cn1:
            #     if ii < ng:
            #         if (self.Kx[ii] != 0.) and (self.Ky[ii] != 0.):
            #             psi0[ii, ii] *= factor
            #             psi0[ii + ng, ii] *= factor
            #             self.phi0[ii, ii] *= factor * 1e-0
            #     else:
            #         iim = ii-ng
            #         if (self.Kx[iim] != 0.) and (self.Ky[iim] != 0.):
            #             psi0[ii-ng, ii] *= factor
            #             psi0[ii, ii] *= factor
            #             self.phi0[ii, ii] *= factor * 1e-0
            #
            # for ii in cn1:
            #     if ii < ng:
            #         if (self.Kx[ii] != 0.) and (self.Ky[ii] != 0.):
            #             psi0[ii, ii] = -1j
            #             psi0[ii + ng, ii] = 1j
            #             self.phi0[ii, ii] = 0.
            #             self.phi0[ii+ng, ii] = 0.
            #
            #     else:
            #         iim = ii-ng
            #         if (self.Kx[iim] != 0.) and (self.Ky[iim] != 0.):
            #             psi0[ii-ng, ii] = 1.j
            #             psi0[ii, ii] = -1.j
            #             self.phi0[ii, ii] = 1.
            #             self.phi0[ii-ng, ii] = -1.

            # # worked for normal incidence, Wood at (\pm 1, \pm 1) orders
            # ii = 0
            # psi0[5+ii, 5+ii] = 1.
            # psi0[5+ii+ng, 5+ii] = -1.
            # self.phi0[5+ii, 5+ii] = 0.
            # self.phi0[5+ii+ng, 5+ii] = 0.
            #
            # ii = 1
            # psi0[5+ii, 5+ii] = 0.
            # psi0[5+ii+ng, 5+ii] = 0.
            # self.phi0[5+ii, 5+ii] = 1.
            # self.phi0[5+ii+ng, 5+ii] = 1.
            #
            # ii = 2
            # psi0[5+ii, 5+ii] = 0.
            # psi0[5+ii+ng, 5+ii] = 0.
            # self.phi0[5+ii, 5+ii] = 1.
            # self.phi0[5+ii+ng, 5+ii] = 1.
            #
            # ii = 3
            # psi0[5+ii, 5+ii] = 0.
            # psi0[5+ii+ng, 5+ii] = 0.
            # self.phi0[5+ii, 5+ii] = 1.
            # self.phi0[5+ii+ng, 5+ii] = -1.
            #
            # ii = 0
            # psi0[5+ii+ng, 5+ii+ng] = 0.
            # psi0[5+ii, 5+ii+ng] = 0.
            # self.phi0[5+ii+ng, 5+ii+ng] = 1.
            # self.phi0[5+ii, 5+ii+ng] = -1.
            #
            # ii = 1
            # psi0[5+ii+ng, 5+ii+ng] = 1.
            # psi0[5+ii, 5+ii+ng] = 1.
            # self.phi0[5+ii+ng, 5+ii+ng] = 0.
            # self.phi0[5+ii, 5+ii+ng] = 0.
            #
            # ii = 2
            # psi0[5+ii+ng, 5+ii+ng] = 1.
            # psi0[5+ii, 5+ii+ng] = 1.
            # self.phi0[5+ii+ng, 5+ii+ng] = 0.
            # self.phi0[5+ii, 5+ii+ng] = 0.
            #
            # ii = 3
            # psi0[5+ii+ng, 5+ii+ng] = 1.
            # psi0[5+ii, 5+ii+ng] = -1.
            # self.phi0[5+ii+ng, 5+ii+ng] = 0.
            # self.phi0[5+ii, 5+ii+ng] = 0.

            self.psi0 = psi0

            # print('_calc_psi0', time.process_time() - t1)

    def _calc_im0(self):
        if self._num_g_ac:
            a0 = np.diag(2 * np.ones(2 * self._num_g_ac))
            b0 = np.zeros((2 * self._num_g_ac, 2 * self._num_g_ac))
            self.im0 = (a0, b0)

    def _calc_s_0(self):
        """calculate vacuum scattering matrix"""
        # todo: change into sparse to save memory
        if self._num_g_ac:
            # t1 = time.process_time()
            s11_0 = np.zeros((2 * self._num_g_ac, 2 * self._num_g_ac), dtype=complex)
            s12_0 = np.eye(2 * self._num_g_ac, dtype=complex)
            s21_0 = s12_0
            s22_0 = s11_0
            self.sm0 = (s11_0, s12_0, s21_0, s22_0)
            # print('_calc_s_0_3d', time.process_time() - t1)

    @property
    def uc_area(self) -> float:
        """Unit cell area"""
        return self._uc_area

    def _calc_uc_area(self):
        if self.latt_vec:
            if type(self.latt_vec) is not tuple:
                self._uc_area = self.latt_vec
            else:
                a1, a2 = self.latt_vec
                a1n, a2n = [la.norm(a) for a in [a1, a2]]
                if a1n != 0. and a2n != 0.:
                    self._uc_area = np.abs(np.cross(self.latt_vec[0], self.latt_vec[1]))
                elif a1n == 0.:  # 1D
                    self._uc_area = a2n
                elif a2n == 0.:
                    self._uc_area = a1n
                else:
                    raise Exception("Both lattice vectors have zero norms. Can't calculate unit cell area.")

    def _calc_angles(self):
        """
        calculate cos(vartheta), sin(vartheta), cos(phi), sin(phi) for all relevant orders, where vartheta = pi/2-theta
        This is used for setting incidence in high orders.
        """
        if self.ks:
            # t1 = time.process_time()

            cphi = np.zeros(self._num_g_ac, complex)
            sphi = np.zeros(self._num_g_ac, complex)
            cthe = np.zeros(self._num_g_ac, complex)
            sthe = np.zeros(self._num_g_ac, complex)

            idxa = np.array(self.idx_g)
            ksa = np.array(self.ks)
            k_pa = la.norm(ksa, axis=-1)  # norm of in-plane momentum

            i0 = (k_pa == 0.)
            ib = (k_pa != 0.)

            cphi[i0] = 1.
            sphi[i0] = 0.
            cphi[ib] = ksa[ib, 0] / k_pa[ib]
            sphi[ib] = ksa[ib, 1] / k_pa[ib]
            cthe = k_pa / self.omega + 0j
            sthe = np.sqrt(1 - cthe**2 + 0j)

            # these ensure the incident channel has real cos, sin
            ii = (idxa[:, 0] == 0) & (idxa[:, 1] == 0)
            cphi[ii] = np.cos(self._phi)
            sphi[ii] = np.sin(self._phi)
            cthe[ii] = np.cos(np.pi/2 - self._theta)
            sthe[ii] = np.sin(np.pi/2 - self._theta)

            self.cos_phis = list(cphi)
            self.sin_phis = list(sphi)
            self.cos_varthetas = list(cthe)
            self.sin_varthetas = list(sthe)

            # print('_calc_angles', time.process_time()-t1)
        self.calc_ai_bo_3d()

    def calc_ai_bo_3d(self):
        """calculate incident ai and bo amplitudes"""

        # t1 = time.process_time()

        o = self.omega

        aibo = [self.ai, self.bo]
        for ii, (sa, pa, od) in enumerate([[self._s_amps, self._p_amps, self._incident_orders], [self._s_amps_bk, self._p_amps_bk, self._incident_orders_bk]]):
            if self._num_g_ac:
                ab = np.zeros(2 * self._num_g_ac) + 0j
                if (sa or pa) and od and self.idx_g and \
                        self.sin_phis and self.sin_varthetas and self.cos_phis and self.cos_varthetas:
                    # find the index of the input orders in the g list
                    idx = [i for order in od for i, j in enumerate(self.idx_g) if j == order]
                    for i, jj in enumerate(idx):
                    # for i in range(len(idx)):
                        if (jj in self.q0_0):
                            # todo: need to handle this and document it.
                            #  if user specify 90 degree incidence, this is activated
                            warn('You are specifying incidence in a channel that is parallel to the surface of the structure. \n In this case, only specific field configuration is allowed.')
                            ab[jj] = sa[i]
                            ab[jj + self._num_g_ac] = pa[i]
                        else:
                            s = sa[i]
                            p = pa[i]

                            # # original
                            # ab[i] = -s * self.sin_phis[i] + p * self.sin_varthetas[i] * self.cos_phis[i]  # e_x
                            # ab[i + self._num_g_ac] = s * self.cos_phis[i] + p * self.sin_varthetas[i] * self.sin_phis[i]  # e_y

                            # # original, corrected
                            # ab[jj] = -s * self.sin_phis[jj] + p * self.sin_varthetas[jj] * self.cos_phis[jj]  # e_x
                            # ab[jj + self._num_g_ac] = s * self.cos_phis[jj] + p * self.sin_varthetas[jj] * self.sin_phis[jj]  # e_y

                            # with new calc that removed convergence problem at Wood
                            ex = -s * self.sin_phis[jj] + p * self.sin_varthetas[jj] * self.cos_phis[jj]  # e_x
                            ey = s * self.cos_phis[jj] + p * self.sin_varthetas[jj] * self.sin_phis[jj]  # e_y
                            phi_2x2 = self.phi0_2x2s[:, :, jj]
                            phi_2x2_i = 1 / la.det(phi_2x2) * np.array([[phi_2x2[1, 1], -phi_2x2[0, 1]],
                                                                        [-phi_2x2[1, 0], phi_2x2[0, 0]]])
                            # v = la.solve(phi_2x2, np.array([ex, ey]))
                            v = phi_2x2_i @ np.array([ex, ey])
                            ab[jj] = v[0]
                            ab[jj + self._num_g_ac] = v[1]

                            # todo: if incident region is not vacuum, here ai is not pr.ai
                            # todo: for uniform non-vacuum incident region, ai should not be pr.ai.
                            # todo: ai bo is calculated with assumption that incident and output are vacuum, with whatever choice of 2x2 eigen. ai bo are amplitudes of corresponding eigens. However, for non vacuum layer, the phil and psil are not the same as that in vacuum, so directly copying ai here is incorrect.

                aibo[ii] = ab
        self.ai, self.bo = aibo
        # print('calc_ai_bo_3d', time.process_time() - t1)

