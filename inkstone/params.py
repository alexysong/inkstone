# -*- coding: utf-8 -*-

###for testing only
import sys
sys.path.append("C:/Users/w-a-c/Desktop/inkstone")
###----------------

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
from GenericBackend import genericBackend as gb




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
                 show_calc_time = False,
                 gb=gb
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

        self.gs: Optional[List[Tuple[float, float]]] = None  # list of g points for E and H fields. Added by k_pa_inci to get the ks, i.e. k points for E and H
        self.idx_g: Optional[List[Tuple[int, int]]] = None  # list of g points indices
        self.idx_conv_mtx: Optional[any] = None  # indexing array to for constructing convolution matrices.
        self.ks: Optional[List[Tuple[float, float]]] = None  # list of k points for E and H fields.
        self.idx_g_ep_mu: Optional[List[Tuple[int, int]]] = None
        self.idxa_g_ep_mu: Optional[Tuple[any, any]] = None  # The g indices for ep and mu.
        self.idx_g_ep_mu_used: Optional[List[Tuple[int, int]]] = None  # the actually used epsilon and mu grid points
        self.ks_ep_mu: Optional[List[Tuple[float, float]]] = None  # list of k points where epsi and mu Fourier components are to be calculated
        self.ka_ep_mu: Optional[Tuple[any, any]] = None  # The k points for ep and mu, Tuple of meshgrid (kx, ky).
        self.Kx: Optional[any] = None  # the diagonals of the Big diagonal matrix
        self.Ky: Optional[any] = None  # the diagonals of the Big diagonal matrix
        self.Kz: Optional[any] = None  # the diagonals of the Big diagonal matrix, useful for uniform layers
        self.mmax: Optional[int] = None  # max index in b1 direction for epsi and mu Fourier series
        self.nmax: Optional[int] = None  # max index in b2 direction for epsi and mu Fourier series
        self.phi0: Optional[any] = None  # E field eigen mode in vacuum (identity matrix), side length 2*num_g
        self.psi0: Optional[any] = None  # H field eigen mode in vacuum (Q * Phi * q^-1), side length 2*num_g
        self.phi0_2x2s: Optional[any] = None  # the non-zero elements of phi0 stored in (2, 2, num_g) shape
        self.phif: Optional[any] = None  # E field eigen mode in fic material, shape (2num_g, 2num_g)
        self.psif: Optional[any] = None  # H field eigen mode in fic material, shape (2num_g, 2num_g)
        self.q0: Optional[any] = None  # 1d array, eigen propagation constant in z direction in vacuum, length 2*num_g
        self.q0_half: Optional[any] = None  # 1d array, eigen propagation constant in z direction in vacuum, length num_g
        self.q0_0: Optional[any] = None  # 1d array, containing idxs to the idx_g list which q0 is 0, i.e. parallel to surface
        self.q0_inv: Optional[any] = None  # 1d array, 1./q0, elementwise inversion of q0, length 2*num_g
        self.P0_val: Optional[Tuple[any, any, any, any]] = None  # Tuple of 4, each is an ndarray of size num_g, containing the diagonal elements of the 4 blocks of P0.
        self.Q0_val: Optional[Tuple[any, any, any, any]] = None  # Tuple of 4, each is an ndarray of size num_g, containing the diagonal elements of the 4 blocks of Q0.
        self.im0: Optional[Tuple[any, any]] = None  # vacuum interface matrix
        self.sm0: Optional[Tuple[any, any, any, any]] = None  # vacuum scattering matrix, (s11_0, s12_0, s21_0, s22_0), each of s_ij has side length 2*num_g
        self._rad_cha_0: Optional[List[int]] = None

        self.ccnif = "physical"
        self.ccpif = "ac"

        self.__num_g_actual: Optional[int] = None  # actual number of G points used.
        self._num_g_input: Optional[int] = None  # number of G points input by user
        self._omega: Optional[complex] = None
        self._k_pa_inci: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None  # incident in-plane k
        self._kii: Optional[Union[float, complex]] = None
        self._kio: Optional[Union[float, complex]] = None
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
        self.cos_varthetas_bk: Optional[List[float]] = None  # incident angles
        self.sin_varthetas_bk: Optional[List[float, complex]] = None
        self.cos_phis_bk: Optional[List[float]] = None
        self.sin_phis_bk: Optional[List[float]] = None

        self._incident_orders = None
        self._incident_orders_bk = None
        self._s_amps = None
        self._p_amps = None
        self._s_amps_bk = None
        self._p_amps_bk = None
        self._ai: Optional[any] = None
        self._bo: Optional[any] = None
        self.iesbe: bool = False
        self.iesbtpsp: bool = False
        self.iesbksp: bool = False

        # self.q0_contain_0: bool = False

        self.show_calc_time = show_calc_time

        self._inci_is_vac: Optional[bool] = None  # these are automatically updated when adding first layer
        self._inci_is_iso_nonvac: Optional[bool] = None
        self._ind_inci: Optional[Union[float, complex]] = None
        self._out_is_vac: Optional[bool] = None
        self._out_is_iso_nonvac: Optional[bool] = None
        self._ind_out: Optional[Union[float, complex]] = None
        
        self.gb = gb

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
    def ai(self):
        warn("`ai` and `bo` are not stored in Params anymore. Please use `Inkstone.ai` and `Inkstone.bo` to retrieve them.")
        return self._ai

    @ai.setter
    def ai(self, val):
        self._ai = val

    @property
    def bo(self):
        warn("`ai` and `bo` are not stored in Params anymore. Please use `Inkstone.ai` and `Inkstone.bo` to retrieve them.")
        return self.bo

    @bo.setter
    def bo(self, val):
        self._bo = val

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
            val = self.gb.parseData(((val, 0), (0, 0)),self.gb.float64)
            self.is_1d_latt = True
        else:
            a, b = val
            an, bn = self.gb.parseData([self.gb.la.norm(a), self.gb.la.norm(b)],dtype=self.gb.float64)
            # if either latt vec is zero, then it is 1D. Make sure it's x-z for layer solver.
            if an == 0:
                val =self.gb.parseData(((bn, 0), (0, 0)),dtype=self.gb.float64)
                self.is_1d_latt = True
                warn("2D structure (in-plane 1D), non-uniform direction is rotated to x axis.")
            if bn == 0:
                val_old = val
                val = self.gb.parseData(((an, 0), (0, 0)),dtype=self.gb.float64)
                self.is_1d_latt = True
                if val_old != val:
                    warn("2D structure (in-plane 1D), non-uniform direction is rotated to x axis.")
        self._latt_vec: Union[Tuple[Tuple[float, float], Tuple[float, float]]] = val
        self._recipr_vec: Tuple[Tuple[float, float], Tuple[float, float]] = recipro(val[0], val[1], gb=self.gb)
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
        # self._calc_ai_bo_3d()  # called through _calc_gs

    @property
    def frequency(self) -> Union[float, complex]:
        """ frequency, c/lambda in vacuum """
        return self._frequency

    @frequency.setter
    def frequency(self, val: Union[float, complex]):
        if val is not None:
            self.omega = val * self.gb.pi * 2.
            self._frequency = val

    @property
    def omega(self) -> Union[float, complex]:
        """ angular frequency, 2pi c/lambda in vacuum """
        return self._omega

    @omega.setter
    def omega(self, val: Union[float, complex]):
        if val is not None:
            self._omega = self.gb.parseData(val, dtype=self.gb.complex128)
            self._frequency = self.gb.parseData(val / self.gb.pi / 2.)
            # self.q0_contain_0 = False
            # self._calc_gs()  # recalculating gs because some g may be possibly removed in previous runs to remove Wood's anomaly.
            self._calc_kii()
            self._calc_kio()
            # self._calc_k_pa_inci()  # called through _calc_ki()
            # self._calc_q0()  # called through _calc_ki() - _calc_k_pa_inci() or _calc__calc_theta_phi_from_k_pa() - _calc_ks.
            # self._calc_P0Q0()  # called through _calc_ki() - _calc_k_pa_inci or _calc__calc_theta_phi_from_k_pa() - _calc_ks - _calc_Km.
            # self._calc_angles()  # called through _calc_ki() - _calc_k_pa_inci  or _calc__calc_theta_phi_from_k_pa() - _calc_ks - _calc_Km.

    @property
    def kii(self) -> Union[float, complex]:
        return self._kii

    @property
    def kio(self) -> Union[float, complex]:
        return self._kio

    def _calc_kii(self):
        """Calculate incident k in inci region"""
        if self.omega is not None:
            o = self.omega

            if (self.inci_is_vac or self.inci_is_iso_nonvac) and (self.ind_inci is not None):
                r = self.ind_inci
                self._kii = o * r
            else:
                self._kii = None

            if self.iesbtpsp:
                self._calc_k_pa_inci()
            elif self.iesbksp or self.iesbe:
                self._calc_theta_phi_from_k_pa()

    def _calc_kio(self):
        """Calculate incident k in out region"""
        if self.omega is not None:
            o = self.omega

            if (self.out_is_vac or self.out_is_iso_nonvac) and (self.ind_out is not None):
                r = self.ind_out
                self._kio = o * r
            else:
                self._kio = None

    @property
    def k_pa_inci(self) -> Tuple[float, float]:
        """kx and ky projection of the incident wave vector"""
        return self._k_pa_inci

    @k_pa_inci.setter
    def k_pa_inci(self, val: Tuple[float]):
        """Directly setting incident in-plane kx and ky. This is external API. Setting theta and phi should not call this."""
        if val is not None:
            self._k_pa_inci = val
            # kx = val[0]
            # ky = val[1]

            self._calc_theta_phi_from_k_pa()

    def _calc_theta_phi_from_k_pa(self):
        if self.k_pa_inci is not None:
            kx, ky = self.k_pa_inci
            if kx == 0. and ky == 0.:
                self._theta = self.gb.dataParser(0.)
                # self.theta = 0.
                if self.phi is None:
                    warn("Both kx and ky are zero, but phi is not specified. Now defaulting to phi = 0.", UserWarning)
                    self._phi = 0.  # not setting property self.pr.phi to avoid double calling stuff like self.pr._calc_ks()
                    # self.phi = 0.
                # else:
                #     warn("Both kx and ky are zero. In this case incident angle phi need to be set explicitly. Using the existing phi value.", UserWarning)
                #     # self._phi = phi / 180. * self.gb.pi
                #     # self.phi = phi
            else:
                kn = self.gb.sqrt(self.gb.abs(kx) ** 2 + self.gb.abs(ky) ** 2)
                self._phi = self.gb.arccos(kx / kn)
                # self.phi = self.gb.arccos(kx / kn) * 180. / self.gb.pi
                if self.kii is not None:
                    self._theta = self.gb.arcsin(kn / self.kii.real)
                    # self.theta = self.gb.arcsin(kn / self.kii.real) * 180. / self.gb.pi

            self._calc_ks()
            # self._calc_angles()  # called through _calc_ks()

    @property
    def theta(self) -> Union[float, complex]:
        """he angle between incident k and z axis, in degrees, range: [0, pi/2]"""
        if self._theta is not None:
            theta = self._theta / self.gb.pi * 180.
        else:
            theta = None
        return theta

    @theta.setter
    def theta(self, val: Union[float, complex]):
        if val is not None:
            self._theta: self.gb.float64 = self.gb.parseData(val * self.gb.pi / 180.)
            self._calc_k_pa_inci()
            # self._calc_angles()  # called through _calc_k_pa_inci() - _calc_ks()
        else:
            # if self._theta is None:
            #     self._theta = 0.
            self._theta = None

    @property
    def phi(self) -> float:
        """ the angle between the in-plane projection of k and the x axis in degrees. Rotating from kx axis phi degrees ccw around z axis to arrive at the kx of incident wave. """
        if self._phi is not None:
            phi = self._phi / self.gb.pi * 180.
        else:
            phi = None
        return phi

    @phi.setter
    def phi(self, val: float):
        if val is not None:
            self._phi: float = self.gb.parseData(val * self.gb.pi / 180.)
            self._calc_k_pa_inci()
            # self._calc_angles()  # called through _calc_k_pa_inci() -  _calc_ks()
        else:
            # if self._phi is None:
            #     self._phi = 0.
            self._phi = None

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
        # Convert order and order_back to lists
        if order is None:
            if self._incident_orders is None:
                order = [(0, 0)]
            else:
                order = self._incident_orders
        elif not hasattr(order, "__len__"):
            order = [(order, 0)]
        elif type(order) is tuple:
            order = [order]
        for od in order:
            if not (od in self.idx_g):
                raise Exception('The incident order you specified is not within the order list. Possible solution is to reduce order or increase the number of G points.')
        self._incident_orders = order
        # considered allowing 1d several orders. But then ambiguity:is [0, 1] 2d order (0, 1) or is it two 1d orders 0 and 1?

        if order_back is None:
            if self._incident_orders_bk is None:
                order_back = [(0, 0)]
            else:
                order_back = self._incident_orders_bk
        elif not hasattr(order_back, "__len__"):
            order_back = [(order, 0)]
        elif type(order_back) is tuple:
            order_back = [order_back]
        for od in order_back:
            if not (od in self.idx_g):
                raise Exception('The incident order you specified is not within the order list. Possible solution is to reduce order or increase the number of G points.')
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

        # self._calc_ai_bo_3d()

    @property
    def inci_is_vac(self) -> bool:
        return self._inci_is_vac

    @inci_is_vac.setter
    def inci_is_vac(self, val: bool):
        if val != self._inci_is_vac:
            self._inci_is_vac = val
            self._calc_kii()
            # self._calc_k_pa_inci()  # called through _calc_ki()

    @property
    def out_is_vac(self) -> bool:
        return self._out_is_vac

    @out_is_vac.setter
    def out_is_vac(self, val: bool):
        if val != self._out_is_vac:
            self._out_is_vac = val
            self._calc_kio()

    @property
    def inci_is_iso_nonvac(self) -> bool:
        """if inci region is isotropic non-vacuum"""
        return self._inci_is_iso_nonvac

    @inci_is_iso_nonvac.setter
    def inci_is_iso_nonvac(self, val: bool):
        if val != self._inci_is_iso_nonvac:
            self._inci_is_iso_nonvac = val
            self._calc_kii()
            # self._calc_k_pa_inci()  # called through _calc_ki()

    @property
    def ind_inci(self) -> float:
        """inci region refractive index"""
        return self._ind_inci

    @ind_inci.setter
    def ind_inci(self, val):
        if val != self._ind_inci:
            self._ind_inci = val
            self._calc_kii()
            # self._calc_k_pa_inci()  # called through _calc_ki()

    @property
    def out_is_iso_nonvac(self) -> bool:
        """if out region is isotropic non-vacuum"""
        return self._out_is_iso_nonvac

    @out_is_iso_nonvac.setter
    def out_is_iso_nonvac(self, val: bool):
        if val != self._out_is_iso_nonvac:
            self._out_is_iso_nonvac = val
            self._calc_kio()

    @property
    def ind_out(self) -> float:
        """out region refractive index"""
        return self._ind_out

    @ind_out.setter
    def ind_out(self, val):
        if val != self._ind_out:
            self._ind_out = val
            self._calc_kio()

    def _calc_k_pa_inci(self):
        """calculate incident kx and ky"""
        if (self._theta is not None) and (self._phi is not None) and (self.kii is not None):
            kx = self.kii.real * self.gb.cos(self.gb.pi/2 - self._theta) * self.gb.cos(self._phi)
            ky = self.kii.real * self.gb.cos(self.gb.pi/2 - self._theta) * self.gb.sin(self._phi)
            # This is where it determines that the (kx, ky) of the structure is determined by the incident region's refractive index, theta and phi, not the output region.

            self._k_pa_inci = [kx, ky]
            self._calc_ks()  # called in `k_pa_inci` setter

            # todo: complex kx ky also gives answers, but the physical meaning is different

    def _calc_gs(self):
        """ calculate E and H Fourier components g points """
        if self._num_g_input and self.recipr_vec:
            b1, b2 = self.recipr_vec
            b1n, b2n = [self.gb.la.norm(b) for b in [b1, b2]]
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
            # self._calc_ai_bo_3d()  # called through _calc_ks() - _calc_angles()

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
        if self.gs and (self.k_pa_inci is not None):
            self.ks = self.gb.parseData([[g[0] + self.k_pa_inci[0], g[1] + self.k_pa_inci[1]] for g in self.gs], dtype=self.gb.float64)
            self._calc_Km()
            self._calc_q0()
            self._calc_ks_ep_mu()
            self._calc_angles()

    def _calc_Km(self):
        """Calculate Kx Ky arrays"""
        if self.ks is not None:
            # t1 = time.process_time()
            ksa = self.gb.parseData(self.ks,dtype=self.gb.float64)  # shape (NumG, 2)
            kx = ksa[:, 0]
            ky = ksa[:, 1]
            self.Kx = kx
            self.Ky = ky

            # print('_calc_Km', time.process_time() - t1)

            self._calc_P0Q0()
            # self._calc_phi0_psi0()  # called through _calc_P0Q0()

    def _calc_conv_mtx_idx(self):
        if self.idx_g:
            self.idx_conv_mtx = conv_mtx_idx_2d(self.idx_g, self.idx_g, gb=self.gb)
            
            reshaped_tensor = self.gb.reshape(self.idx_conv_mtx, [self._num_g_ac ** 2, 2])
            tuple_list = [(int(i), int(j)) for i, j in reshaped_tensor]
            self.idx_g_ep_mu_used = list(set(tuple_list))
           # self.idx_g_ep_mu_used = list(set([(i, j) for (i, j) in self.idx_conv_mtx.reshape(self._num_g_ac ** 2, 2)]))  # The first element of each tuple is in physical "x" direction

    def _calc_ks_ep_mu(self):
        if self.idx_g:
            # t1 = time.process_time()

            m, n = max_idx_diff(self.idx_g,self.gb)
            self.mmax = m
            self.nmax = n
            x = self.gb.arange(-m, m+1)
            y = self.gb.arange(-n, n+1)
            xx, yy = self.gb.meshgrid(x, y)

            # t2 = time.process_time()

            self.idx_g_ep_mu = list(zip(xx.ravel(), yy.ravel()))
            self.idxa_g_ep_mu = (xx, yy)

            # t3 = time.process_time()

            b1, b2 = self.recipr_vec
            b1n, b2n = [self.gb.la.norm(b) for b in [b1, b2]]
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
        if None not in [self._num_g_ac,self.omega,self.ks]:
            # t1 = time.process_time()
            k_parallel = self.gb.la.norm(self.ks, axis=-1)
            q02 = self.gb.ones(self._num_g_ac) * self.gb.square(self.omega) - self.gb.square(k_parallel) + 0j
            self._rad_cha_0 = self.gb.where(q02.real > 0)[0].tolist()  # todo: even for radiation channel, if omega.imag larger than omega.real, q02.real is negative
            q0 = self.gb.sqrt(q02)
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

            self.q0_0 = self.gb.where(self.gb.abs(q0) == 0.)[0]

            self.q0 = self.gb.concatenate([q0, q0])
            self.q0_half = q0
            # print('_calc_q0', time.process_time() - t1)

         #   with self.gb.errstate(divide='ignore', invalid='ignore'):
            self.q0_inv = 1. / self.q0 #seems default behavior is set to inf
            #ii = self.gb.where(self.q0 == 0.)
            #self.q0_inv[ii] = float('inf')

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
            # self.P0 = self.gb.zeros((2*ng, 2*ng), dtype=self.gb.complex128)
            # r1 = range(ng)
            # r2 = range(ng, 2 * ng)
            # self.P0[r1, r1] = P11
            # self.P0[r2, r1] = P21
            # self.P0[r1, r2] = P12
            # self.P0[r2, r2] = P22
            # self.P0 = self.gb.block([[self.gb.diag(P11), self.gb.diag(P12)],
            #                     [self.gb.diag(P21), self.gb.diag(P22)]])

            # self._calc_psi0()
            # self._calc_phi0_psi0()  # logically, should call this, however, the only place calling _calc_P0Q0() is _calc_Km(), which is only called at _calc_ks(). There, after _calc_Km(), _calc_q0() is called, which calls this.
            # self._calc_phif_psif()

    def _calc_phi0_psi0(self):
        if (self.Q0_val is not None) and (self.q0 is not None) and (self._num_g_ac is not None):
            # t1 = time.process_time()

            # o = self.omega
            # Kx = self.Kx.copy()
            # Ky = self.Ky.copy()
            # k_norm = self.gb.sqrt(self.gb.conj(Kx) * Kx + self.gb.conj(Ky) * Ky)


            # # Wood stable as eigen
            # Tx = Kx  # term with x subscript
            # Ty = Ky
            #
            # # for small k just use [1, 0], [0, 1] as eigen.
            # skc = 0.5  # small k criterion
            # small_k = self.gb.where(k_norm < (skc * self.gb.abs(o)))[0]
            # Tx[small_k] = 0.
            # Ty[small_k] = 1.
            # t_norm = k_norm.copy()  # term norm
            # t_norm[small_k] = 1.
            #
            # Txm = self.gb.diag(Tx)
            # Tym = self.gb.diag(Ty)
            # jwTxq = 1j / o * Tx * self.q0_half
            # jwTyq = 1j / o * Ty * self.q0_half
            #
            # # for small k, need separate calc here
            # jwTxq[small_k] = -1j / o * Kx[small_k] * Ky[small_k] / self.q0_half[small_k]
            # jwTyq[small_k] = -1j / o * (-self.gb.conj(Kx[small_k])*Kx[small_k] - self.gb.conj(self.q0_half[small_k])*self.q0_half[small_k]) / self.q0_half[small_k]
            #
            # jwTxqm = self.gb.diag(jwTxq)
            # jwTyqm = self.gb.diag(jwTyq)
            #
            # ng = self._num_g_ac
            # r1 = range(ng)
            # r2 = range(ng, 2 * ng)
            # phi0 = self.gb.zeros((2*ng, 2*ng), dtype=self.gb.complex128)
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
            # # phi0 = self.gb.block([[Tym, jwTxqm],
            # #                  [-Txm, jwTyqm]])
            # # psi0 = self.gb.block([[jwTxqm, Tym],
            # #                  [jwTyqm, -Txm]])
            #
            # self.phi0_2x2s = self.gb.inputParser([[Ty, jwTxq],
            #                           [-Tx, jwTyq]])


            # with normalization
            q0h = self.q0_half
            ng = self._num_g_ac
            o = self.omega

            ksa = self.gb.parseData(self.ks)
            Kx = ksa[:, 0]
            Ky = ksa[:, 1]
            # Kx = self.Kx.copy()
            # Ky = self.Ky.copy()
            k_norm = self.gb.sqrt(self.gb.conj(Kx) * Kx + self.gb.conj(Ky) * Ky)

            # skc = 0.05
            # i_knz = self.gb.where(k_norm < (skc * self.gb.abs(o)))[0]
            i_kez = self.gb.where(k_norm == 0.)[0]
            # i_qsw = self.gb.where((self.gb.abs(q0h) <= self.gb.abs(o)) * (k_norm >= (skc * self.gb.abs(o))))[0]
            i_qsw = self.gb.where((self.gb.abs(q0h) < self.gb.abs(o)))[0]
            i_qlw = self.gb.where(self.gb.abs(q0h) > self.gb.abs(o))[0]
            # when allowing imaginary/complex kx, ky: when ky = i kx, q = omega, but k_norm is not zero. But this case doesn't matter. as long as k_norm not zero, it can be divided.
            idxa = self.gb.parseData(self.idx_g)
            ii = (idxa[:, 0] == 0) & (idxa[:, 1] == 0)

            c1 = self.gb.concat([Ky, -Kx])
            c2 = self.gb.concat([Kx, Ky])
            c1f = self.gb.ones(ng, dtype=self.gb.complex128)
            c2f = self.gb.clone(c1f)

            # c1[:, i_knz] = self.gb.inputParser([[1.], [0.]])
            # c2[:, i_knz] = -1j / o * self.gb.inputParser([Kx[i_knz] * Ky[i_knz] / q0h[i_knz], (-self.gb.square(Kx[i_knz]) - self.gb.square(q0h[i_knz])) / q0h[i_knz]])  # should not be |Kx|^2
            c1[:, i_kez] = self.gb.parseData([[1.], [0.]], dtype=self.gb.complex128)
            c2[:, i_kez] = self.gb.parseData([[0.], [1.]], dtype=self.gb.complex128)
            cphi = self.gb.cos(self._phi)
            sphi = self.gb.sin(self._phi)
            c1[:, ii] = self.gb.parseData([[sphi], [-cphi]], dtype=self.gb.complex128)
            c2[:, ii] = self.gb.parseData([[cphi], [sphi]], dtype=self.gb.complex128)

            c1f[i_qlw] = o / q0h[i_qlw] / k_norm[i_qlw]
            c2f[i_qlw] = 1j / k_norm[i_qlw]

            c1f[i_qsw] = 1. / k_norm[i_qsw]
            c2f[i_qsw] = 1j / o * q0h[i_qsw] / k_norm[i_qsw]

            # c1f[i_knz] = 1.
            # c2f[i_knz] = 1.

            c1f[i_kez] = 1.
            c2f[i_kez] = 1j * q0h[i_kez] / o

            c1f[ii] = 1.
            c2f[ii] = 1j * q0h[ii] / o

            c1 *= c1f
            c2 *= c2f

            r1 = range(ng)
            r2 = range(ng, 2 * ng)
            phi0 = self.gb.zeros((2*ng, 2*ng), dtype=self.gb.complex128)
            psi0 = phi0.copy()
            phi0[r1, r1] = c1[0, :]
            phi0[r2, r1] = c1[1, :]
            phi0[r1, r2] = c2[0, :]
            phi0[r2, r2] = c2[1, :]
            psi0[r1, r1] = c2[0, :]
            psi0[r2, r1] = c2[1, :]
            psi0[r1, r2] = c1[0, :]
            psi0[r2, r2] = c1[1, :]

            self.phi0_2x2s = self.gb.moveaxis(self.gb.parseData([c1, c2]), 0, 1)

            # # debugging,  check if phi is eigen and consistent with psi
            # psi00 = -1j * self.P0 @ phi0 / self.q0
            # diff = self.gb.abs(psi00 - psi0).max()
            # print('psi0 diff {:g}'.format(diff))
            #
            # check_eigen = self.P0 @ self.P0 @ phi0
            # diff1 = self.gb.abs(check_eigen + phi0 * self.q0 * self.q0).max()
            # print('check eigen {:g}'.format(diff1))

            # # original
            # phi0 = self.gb.eye(2 * self._num_g_ac, 2 * self._num_g_ac, dtype=self.gb.complex128)
            # ng = self._num_g_ac
            # psi0 = self.gb.zeros((2 * ng, 2 * ng), dtype=self.gb.complex128)
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

            phif = self.gb.eye(2 * ng, 2 * ng, dtype=self.gb.complex128)
            psif = self.gb.zeros((2 * ng, 2 * ng), dtype=self.gb.complex128)

            r1 = range(ng)
            r2 = range(ng, 2 * ng)

            # attention! phif is assumed to be identity in interface calculations. if need to change the form, you need to change coding there, not just changing phif here.
            # phif[r2, r2] = -1.

            # attention! If need to change this form, note in interface matrix calculations this form is assumed to speed up things. Need to change coding there, not just changing psif here.
            psif[r2, r1] = 1.j
            psif[r1, r2] = -1.j

            self.phif = phif
            self.psif = psif

            # # for debugging
            # phif2 = self.gb.eye(2 * ng, 2 * ng, dtype=self.gb.complex128)
            # psif2 = self.gb.zeros((2 * ng, 2 * ng), dtype=self.gb.complex128)
            # phif2[r2, r2] = -1.  # attention! this was for debugging, but note since phif=eye, it is omitted in certain calculations
            # psif2[r2, r1] = 1.
            # psif2[r1, r2] = -1.
            # self.phif2 = phif2
            # self.psif2 = psif2

            # phif3 = self.gb.zeros((2 * ng, 2 * ng), dtype=self.gb.complex128)
            # psif3 = self.gb.zeros((2 * ng, 2 * ng), dtype=self.gb.complex128)
            # phif3[r1, r1] = 1.
            # phif3[r2, r1] = 1j
            # phif3[r1, r2] = -1j
            # phif3[r2, r2] = 1.
            # psif3 = phif3
            #
            # self.phif3 = phif3
            # self.psif3 = psif3

            # if self.P0_val is not None:
            #     phif4 = self.gb.random.rand(2*ng, 2*ng)
            #     psif4 = -1j * self.P0 @ phif4 * self.q0_inv
            #     self.phif4 = phif4
            #     self.psif4 = psif4

            # # end of debugging

    def _calc_phi0(self):
        warn('This method is deprecated. Use `_calc_phi0_psi0` instead.', category=DeprecationWarning)
        if self._num_g_ac:
            self.phi0 = self.gb.eye(2 * self._num_g_ac, 2 * self._num_g_ac, dtype=self.gb.complex128)

    def _calc_psi0(self):
        warn('This method is deprecated. Use `_calc_phi0_psi0` instead.', category=DeprecationWarning)

        if (self.Q0_val is not None) and (self.q0 is not None) and (self._num_g_ac is not None) and ((self.Kx is not None) and (self.Ky is not None)):
            # t1 = time.process_time()

            # if not self.q0_contain_0:
            ng = self._num_g_ac
            psi0 = self.gb.zeros((2 * ng, 2 * ng), dtype=self.gb.complex128)
            r1 = range(ng)
            r2 = range(ng, 2 * ng)
            q0_inv = self.q0_inv
            if self.Q0_val[0].size == self.q0_inv[:ng].size:
                psi0[r1, r1] = -1j * self.Q0_val[0] * q0_inv[:ng]
                psi0[r1, r2] = -1j * self.Q0_val[1] * q0_inv[ng:]
                psi0[r2, r1] = -1j * self.Q0_val[2] * q0_inv[:ng]
                psi0[r2, r2] = -1j * self.Q0_val[3] * q0_inv[ng:]

                # # at Wood P is singular so this solving doesn't work.
                # _P = [self.gb.diag(Qv) for Qv in self.Q0_val]
                # P = self.gb.block([[_P[0], _P[1]],
                #               [_P[2], _P[3]]])
                # psi0 = 1j * la.solve(P, self.phi0) @ self.gb.diag(self.q0)

            o = self.omega
            cn = self.gb.where(self.gb.abs(self.q0) == 0.)[0]
            cn1 = self.gb.where(self.gb.abs(self.q0) < 1e-2)[0]  # for debugging

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
            #             psi0[ii, ii] = 1. * self.gb.sign(self.Ky[ii])
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
            #             psi0[ii, ii] = 1. * self.gb.sign(self.Kx[iim])
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
            a0 = self.gb.diag(2 * self.gb.ones(2 * self._num_g_ac))
            b0 = self.gb.zeros((2 * self._num_g_ac, 2 * self._num_g_ac))
            self.im0 = (a0, b0)

    def _calc_s_0(self):
        """calculate vacuum scattering matrix"""
        # todo: change into sparse to save memory
        if self._num_g_ac:
            # t1 = time.process_time()
            s11_0 = self.gb.zeros((2 * self._num_g_ac, 2 * self._num_g_ac), dtype=self.gb.complex128)
            s12_0 = self.gb.eye(2 * self._num_g_ac, dtype=self.gb.complex128)
            s21_0 = s12_0
            s22_0 = s11_0
            self.sm0 = (s11_0, s12_0, s21_0, s22_0)
            # print('_calc_s_0_3d', time.process_time() - t1)

    @property
    def uc_area(self) -> float:
        """Unit cell area"""
        return self._uc_area

    def _calc_uc_area(self):
        if type(self.latt_vec) is not None or self.latt_vec is not None:
            #print(type(self.latt_vec))
            if not isinstance(self.latt_vec, (tuple, self.gb.raw_type)):
                self._uc_area = self.latt_vec
            else:
                a1, a2 = self.latt_vec
                a1n, a2n = [self.gb.la.norm(a) for a in [a1, a2]]
                if a1n != 0. and a2n != 0.:
                    self._uc_area = self.gb.abs(self.gb.cross(self.latt_vec[0], self.latt_vec[1]))
                elif a1n == 0.:  # 1D
                    self._uc_area = a2n
                elif a2n == 0.:
                    self._uc_area = a1n
                else:
                    raise Exception("Both lattice vectors have zero norms. Can't calculate unit cell area.")

    def _calc_angles(self):
        """
        calculate cos(vartheta), sin(vartheta), cos(phi), sin(phi) for all relevant orders, where vartheta = pi/2-theta
        Used in calculating ai bo. Recording these cos and sin allow for high-order incidence.
        """
        if self.ks and self.num_g and (self._theta is not None) and (self._phi is not None) and (self.kii is not None):
            # t1 = time.process_time()

            idxa = self.gb.parseData(self.idx_g)
            ksa = self.gb.parseData(self.ks)
            k_pa = self.gb.la.norm(ksa, axis=-1)  # norm of in-plane momentum
            i0 = (k_pa == 0.)
            ib = (k_pa != 0.)
            ii = (idxa[:, 0] == 0) & (idxa[:, 1] == 0)

            cphi = self.gb.zeros(self._num_g_ac, complex)
            sphi = self.gb.zeros(self._num_g_ac, complex)
            cphi[ib] = ksa[ib, 0] / k_pa[ib]
            sphi[ib] = ksa[ib, 1] / k_pa[ib]
            cphi[i0] = 1.
            sphi[i0] = 0.
            cphi[ii] = self.gb.cos(self._phi)
            sphi[ii] = self.gb.sin(self._phi)

            cthe = k_pa / self.kii.real + 0j  # this is always positive
            # when omega complex, k_parallel is still calc as real
            sthe = self.gb.sqrt(1 - cthe**2 + 0j)
            cthe[ii] = self.gb.cos(self.gb.pi/2 - self._theta)
            sthe[ii] = self.gb.sin(self.gb.pi/2 - self._theta)
            # todo: theta could be None (e.g. setting Excitation By Eigen)

            self.cos_phis = list(cphi)
            self.sin_phis = list(sphi)
            self.cos_varthetas = list(cthe)
            self.sin_varthetas = list(sthe)

            if self.kio is not None:
                cthe_bk = k_pa / self.kio.real + 0j  # this is always positive
                # todo: self.kio is updated in solving stage as which layer is output is determined then. But all `Params` data should be calculated at setting structure stage.
                # when omega complex, k_parallel is still calc as real
                sthe_bk = self.gb.sqrt(1 - cthe_bk ** 2 + 0j)
                cthe_bk[ii] = self.gb.cos(self.gb.pi / 2 - self._theta)  # incorrect, out region could have different refractive index
                sthe_bk[ii] = self.gb.sin(self.gb.pi / 2 - self._theta)
                self.cos_phis_bk = self.cos_phis.copy()
                self.sin_phis_bk = self.sin_phis.copy()
                self.cos_varthetas_bk = list(cthe)
                self.sin_varthetas_bk = list(sthe)

            # print('_calc_angles', time.process_time()-t1)
        # self._calc_ai_bo_3d()

    def _calc_ai_bo_3d(self):
        """calculate incident ai and bo amplitudes"""
        warn('this method is deprecated. It is moved to simulator', DeprecationWarning)

        # t1 = time.process_time()

        o = self.omega

        aibo = [self.ai, self.bo]
        for ii, (sa, pa, od) in enumerate([[self._s_amps, self._p_amps, self._incident_orders], [self._s_amps_bk, self._p_amps_bk, self._incident_orders_bk]]):
            if self._num_g_ac:
                ab = self.gb.zeros(2 * self._num_g_ac) + 0j
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
                            phi_2x2_i = 1 / self.gb.la.det(phi_2x2) * self.gb.parseData([[phi_2x2[1, 1], -phi_2x2[0, 1]],
                                                                        [-phi_2x2[1, 0], phi_2x2[0, 0]]])
                            # v = la.solve(phi_2x2, self.gb.inputParser([ex, ey]))
                            v = phi_2x2_i @ self.gb.parseData([ex, ey])
                            ab[jj] = v[0]
                            ab[jj + self._num_g_ac] = v[1]

                aibo[ii] = ab
        self.ai, self.bo = aibo
        # print('_calc_ai_bo_3d', time.process_time() - t1)

