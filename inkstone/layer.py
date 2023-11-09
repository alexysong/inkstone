# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la
# import scipy.sparse as sps
import scipy.fft as fft
import scipy.linalg as sla
from warnings import warn
from collections import OrderedDict
from typing import Optional, Set, Union, Tuple, Dict, List
import time

from inkstone.ft.ft_2d_cnst import ft_2d_cnst
from inkstone.im import im
from inkstone.sm import s_1l, s_1l_rsp, s_1l_1212, s_1l_1221, s_1l_rsp_lrd
from inkstone.params import Params
from inkstone.bx import Bx
from inkstone.mtr import Mtr
from inkstone.shps import Rect, Para, Elli, Disk, Poly, OneD
from inkstone.helpers.pt_in_poly import pt_in_poly


class Layer:
    # todo: for uniform layer should use sparse matrices, faster less memory

    def __init__(self, name, thickness, material_bg, materials, params, **kwargs):
        """
        A layer.

        One can add patterns to the layer. The layer calculates the fourier series and solve for the scattering matrix of the layer.

        Parameters
        ----------
        name        :   str
                        the name of the layer
        thickness   :   float
                        thickness of the layer. for input (reflected side) region or output (transmitted side) region, the thickness has to be zero.
        material_bg :   str
        materials   :   dict[str, Mtr]
                        all of the materials used in the entire structure.
        params      :   Params
        kwargs      :
                        keyword arguments to pass on to SetLayer

        """

        self.pr: Params = params

        self.if_mod: bool = True  # if this layer is modified
        self._if_t_change: bool = True  # if thickness changed
        self.need_recalc_al_bl = True  # if the al and bl of this layer needs recalculation
        self.is_vac = None
        self.is_diagonal = None
        self.is_isotropic = None
        self.is_dege = None

        self.name: str = name
        self.is_copy = False
        self.original_layer_name = name

        self._thickness: float = thickness
        self.thickness: float = thickness

        self.materials: Dict[str, Mtr] = materials  # all of the materials used in the structure

        self.materials_used: Set[str] = set()  # all the materials used by this layer
        self.materials_used.add(material_bg)
        mb = self.materials[material_bg]
        if mb.epsi[0, 1] != 0 or mb.epsi[1, 0] != 0. or mb.epsi[0, 1] != 0. or mb.mu[0, 1] != 0. or mb.mu[1, 0] != 0.:
            self.materials_ode: bool = True  # if any of the materials used have off-diagonal components
        else:
            self.materials_ode = False

        self.patterns: OrderedDict[str, Bx] = OrderedDict()  # all patterns of this layer

        # for 2d and 3d
        self.epsi_fs: Optional[np.ndarray] = None  # Fourier series components of epsilon in the layer, shape (2mmax+1, 2nmax+1, 3, 3) complex, not ifftshifted
        self.epsi_inv_fs: Optional[np.ndarray] = None
        #
        self.mu_fs: Optional[np.ndarray] = None  # shape (2mmax+1, 2nmax+1, 3, 3) complex, not ifftshifted
        self.mu_inv_fs: Optional[np.ndarray] = None
        #
        # the components that are actually used in convolution matrices
        self.epsi_fs_used: Optional[List[np.ndarray]] = None  # each element is (3, 3) shape
        self.epsi_inv_fs_used: Optional[List[np.ndarray]] = None
        self.mu_fs_used: Optional[List[np.ndarray]] = None
        self.mu_inv_fs_used: Optional[List[np.ndarray]] = None

        # for 3d epsilon and mu convolution matrices
        self.epxxcm, self.epxycm, self.epyxcm, self.epyycm, self.epzzcm, \
        self.eixxcm, self.eixycm, self.eiyxcm, self.eiyycm, self.eizzcm, \
        self.muxxcm, self.muxycm, self.muyxcm, self.muyycm, self.muzzcm, \
        self.mixxcm, self.mixycm, self.miyxcm, self.miyycm, self.mizzcm, \
            = [None for i in range(20)]  # shape （num_g, num_g）

        # --- variables require solving of layer ---

        self.ql: Optional[np.ndarray] = None  # 1d array complex
        self.phil: Optional[np.ndarray] = None  # shape (2num_g, 2num_g)
        self.phil_2x2s: Optional[np.ndarray] = None  # shape (2, 2, num_g)
        self._phil_is_idt = False  # whether phil is identity matrix
        self.psil: Optional[np.ndarray] = None

        self.iml0: Optional[Tuple[np.ndarray, np.ndarray]] = None  # interface matrix
        self.imfl: Optional[Tuple[np.ndarray, np.ndarray]] = None  # interface matrix
        self.sm: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None  # scattering matrix
        self.csm: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None  # cumulative scattering matrix
        self.csmr: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None  # cumulative scattering matrix reversed

        self.al_bl: Optional[Tuple[np.ndarray, np.ndarray]] = None  # the field coefficients (al, bl).

        # --- additional variables and properties ---
        self._in_mid_out: str = 'mid'  # {'in', 'mid', 'out'}, if this layer is the incident, output, or a middle layer

        self._rad_cha: Optional[List[int]] = None

        self._material_bg: Optional[str] = None  # background material
        self.material_bg: str = material_bg  # background material

        self.set_layer(**kwargs)
        # self._set_pr_inci_out()  # called in set_layer() - setting self.material_bg

    @property
    def material_bg(self):
        return self._material_bg

    @material_bg.setter
    def material_bg(self, val):
        if (val is not None) and (self._material_bg != val):
            self._material_bg = val
            self.if_mod = True

            nn = True
            if self.patterns:
                for pattern in self.patterns.values():
                    if pattern.mtr.name != val:
                        self.is_isotropic = False
                        self.is_diagonal = False
                        self.is_dege = False
                        self.is_vac = False
                        nn = False
                        if self.in_mid_out == 'in':
                            self.pr.inci_is_vac = False
                            self.pr.inci_is_iso_nonvac = False
                            self.pr.ind_inci = None
                        break

            if nn:
                mbg: Mtr = self.materials[self._material_bg]

                if mbg.is_isotropic:
                    self.is_isotropic = True
                else:
                    self.is_isotropic = False

                if mbg.is_dege:
                    self.is_dege = True
                else:
                    self.is_dege = False

                if mbg.is_diagonal:
                    self.is_diagonal = True
                else:
                    self.is_diagonal = False

                if mbg.is_vac:
                    self.is_vac = True
                else:
                    self.is_vac = False

            self._set_pr_inci_out()

    @property
    def in_mid_out(self):
        return self._in_mid_out

    @in_mid_out.setter
    def in_mid_out(self, val: str):
        self._in_mid_out = val
        self._set_pr_inci_out()

    def _set_pr_inci_out(self):
        """set inci parameters in Params"""
        if self.in_mid_out == 'in':
            if self.is_vac:
                self.pr.inci_is_vac = True
                self.pr.inci_is_iso_nonvac = False
                self.pr.ind_inci = 1.
            elif self.is_isotropic:
                self.pr.inci_is_vac = False
                self.pr.inci_is_iso_nonvac = True
                if self.materials and self.material_bg:
                    mbg = self.materials[self.material_bg]
                    self.pr.ind_inci = np.sqrt(mbg.epsi[0, 0] * mbg.mu[0, 0])
            else:
                self.pr.inci_is_vac = False
                self.pr.inci_is_iso_nonvac = False
                self.pr.ind_inci = None
        if self.in_mid_out == 'out':
            if self.is_vac:
                self.pr.out_is_vac = True
                self.pr.out_is_iso_nonvac = False
                self.pr.ind_out = 1.
            elif self.is_isotropic:
                self.pr.out_is_vac = False
                self.pr.out_is_iso_nonvac = True
                if self.materials and self.material_bg:
                    mbg = self.materials[self.material_bg]
                    self.pr.ind_out = np.sqrt(mbg.epsi[0, 0] * mbg.mu[0, 0])
            else:
                self.pr.out_is_vac = False
                self.pr.out_is_iso_nonvac = False
                self.pr.ind_out = None

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, val):
        self.set_layer(thickness=val)

    @property
    def rad_cha(self):
        """List of indices in `.params.Params.idx_g` that corresponds to radiation channels in this layer"""
        # For 1 channel with two polarizations this would be [0, num_g]
        # if self.is_vac:
        #     return self.pr.rad_cha_0
        # else:
        return self._rad_cha

    def set_layer(self,
                  thickness: float = None,
                  material_bg: str = None
                  ):
        """
        Set and reset parameters of the layer.
        """
        if (thickness is not None) and (self._thickness != thickness):
            self._thickness = thickness
            self._if_t_change = True

        if material_bg is not None:
            self.material_bg = material_bg

    def add_box(self,
                mtr: str,
                shp: str,
                box_name: str = None,
                **kwargs):
        """
        Add a box to the layer.

        Parameters
        ----------
        mtr         :
        shp         :
        box_name    :
        kwargs      :
                        other parameters to pass on to `~.bx.Bx`
        """
        self.materials_used.add(mtr)
        mt = self.materials[mtr]
        if mt.epsi[0, 1] != 0 or mt.epsi[1, 0] != 0. or mt.epsi[0, 1] != 0. or mt.mu[0, 1] != 0. or mt.mu[1, 0] != 0.:
            self.materials_ode: bool = True  # if any of the materials used have off-diagonal components

        if box_name is None:
            box_name = 'box{:d}'.format(len(self.patterns))
        if box_name in self.patterns:
            warn('A box with this name already exists in this layer. The existing box will be overridden.', UserWarning)
            # todo: remove the material used in the overridden box. be careful not to remove materials still in use by other boxes

        bx = Bx(mt, shp, name=box_name, **kwargs)
        bx.ks = self.pr.ks_ep_mu
        self.patterns[box_name] = bx
        self._find_bx_outside()
        self.if_mod = True

        if not self.materials[mtr].is_vac:
            self.is_vac = False
        if mtr != self.material_bg:
            self.is_isotropic = False
            self.is_diagonal = False
            self.is_dege = False

        self._set_pr_inci_out()

    def set_box(self,
                box_name: str,
                **kwargs):
        """
        set the parameters of an existing box

        Parameters
        ----------
        box_name        :
        kwargs          :
                            shape parameters to reset for box, to be passed on to `.bx.Bx.set_shape`

        Returns
        -------

        """
        if box_name in self.patterns:
            box: Bx = self.patterns[box_name]
            box.set_shape(**kwargs)
            self._find_bx_outside()
            self.if_mod = True
            # todo: allow change the type of material of the box
        else:
            warn('Did not find the name of the box you want to change.', UserWarning)

    def _find_bx_outside(self):
        """ find out which box is in which """

        bxs = list(self.patterns.values())

        bx_areas = np.array([a.shp.area for a in bxs])
        idx = np.argsort(bx_areas)
        bx_names = np.array([a.name for a in bxs])
        bx_name_sorted = bx_names[idx]

        # a fictional bx with background material of this layer.
        bxf = Bx(self.materials[self.material_bg], 'polygon', name='the cell', vertices=[(0, 0), (1, 0), (0, 1)])

        if self.pr.is_1d_latt:
            for ii, name1 in enumerate(bx_name_sorted):
                bx1 = self.patterns[name1]

                # default outside is the fictional bx
                bx1.outside = bxf

                pt1: float = bx1.shp.center

                for name2 in bx_name_sorted[ii + 1:]:
                    bx2 = self.patterns[name2]

                    w2 = bx2.shp.width
                    c2 = bx2.shp.center
                    if c2 - w2 / 2. < pt1 < c2 + w2 / 2.:
                        bx1.outside = bx2
                        break
        else:  # 3D
            for ii, name1 in enumerate(bx_name_sorted):
                bx1 = self.patterns[name1]
                # default outside is the fictional bx
                bx1.outside = bxf

                # find a pt1 that's inside bx1
                if bx1.shp.shape == 'rectangle' or bx1.shp.shape == 'disk' or bx1.shp.shape == 'ellipse' or bx1.shp.shape== 'parallelogram':
                    pt1: Tuple[float, float] = bx1.shp.center
                elif bx1.shp.shape == 'polygon':
                    vertices = bx1.shp.vertices
                    v1, v2, v3 = np.array([vertices[i] for i in range(3)])
                    vm = (v3 + v1) / 2.
                    r = vm - v2
                    while True:
                        r /= -2.
                        v_new = r + v2
                        if pt_in_poly(vertices, v_new):
                            break
                    pt1: Tuple[float, float] = (v_new[0], v_new[1])
                else:
                    raise Exception("Shape of pattern not recognized.")

                for name2 in bx_name_sorted[ii + 1:]:
                    bx2 = self.patterns[name2]

                    # see if pt1 is inside bx2
                    if bx2.shp.shape == 'rectangle':
                        r = np.array(pt1) - np.array(bx2.shp.center)
                        a = bx2.shp.angle * np.pi / 180.
                        r1 = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) @ r
                        if (r1[0] < bx2.shp.side_lengths[0] / 2.) and (r1[1] < bx2.shp.side_lengths[1] / 2.):
                            bx1.outside = bx2
                            break
                    elif bx2.shp.shape == 'parallelogram':
                        shp: Para = bx2.shp
                        r = np.array(pt1) - np.array(shp.center)
                        a = shp.angle * np.pi / 180.
                        rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
                        m = np.tan(np.pi / 2 - shp.shear_angle * np.pi / 180)
                        sheer = np.array([[1, -m],
                                          [0, 1]])
                        r1 = sheer @ rot @ r
                        if (np.abs(r1[0]) < shp.side_lengths[0] / 2.) and (np.abs(r1[1]) < shp.side_lengths[1] * np.sin(shp.shear_angle * np.pi / 180) / 2.):
                            bx1.outside = bx2
                            break
                    elif bx2.shp.shape == 'disk':
                        r = np.array(pt1) - np.array(bx2.shp.center)
                        d = np.linalg.norm(r)
                        if d < bx2.shp.radius:
                            bx1.outside = bx2
                            break
                    elif bx2.shp.shape == 'ellipse':
                        r = np.array(pt1) - np.array(bx2.shp.center)
                        a = bx2.shp.angle * np.pi / 180.
                        r1 = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) @ r
                        if ((r1[0] ** 2 / bx2.shp.half_widths[0] ** 2 + r1[1] ** 2 / bx2.shp.half_widths[1] ** 2) < 1):
                            bx1.outside = bx2
                            break
                    elif bx2.shp.shape == 'polygon':
                        vs = bx2.shp.vertices
                        if pt_in_poly(vs, pt1):
                            bx1.outside = bx2
                            break
                    else:
                        raise Exception("Shape of pattern not recognized.")

    def _calc_ep_mu_fs_3d(self):
        """ calculate the fourier coefficients of this layer """
        # todo: implement add material permittivity and permeability complex tensors
        # todo: for uniform layer, is it simpler?

        t1 = time.process_time()

        # calculate the Fourier components of the background material.
        mtr = self.materials[self.material_bg]
        epsi_bg, epsi_bg_inv, mu_bg, mu_bg_inv = [mtr.epsi, mtr.epsi_inv, mtr.mu, mtr.mu_inv]  # complex 3x3 tensors
        d = np.array(ft_2d_cnst(self.pr.ks_ep_mu), dtype=complex)

        ep, ei, mu, mi = [t[None, :, :] * d[:, None, None] for t in [epsi_bg, epsi_bg_inv, mu_bg, mu_bg_inv]]   # each is complex ((2mmax+1)x(2nmax+1), 3, 3) shape

        for bx in self.patterns.values():
            eb, eib, mb, mib = [np.array(f, dtype=complex) / self.pr.uc_area for f in bx.ft(self.pr.ks_ep_mu)]
            ep += eb
            ei += eib
            mu += mb
            mi += mib

        d1 = 2 * self.pr.mmax + 1
        d2 = 2 * self.pr.nmax + 1
        epa, eia, mua, mia = [a.reshape(d2, d1, 3, 3) for a in [ep, ei, mu, mi]]  # complex ((2nmax+1), (2mmax+1), , 3, 3) shape
        self.epsi_fs = epa
        self.epsi_inv_fs = eia
        self.mu_fs = mua
        self.mu_inv_fs = mia

        if self.pr.show_calc_time:
            print("{:.6f}   _calc_ep_mu_fs_3d".format(time.process_time() - t1) + ", layer "+self.name)

    def _cons_ep_mu_cm_3d(self):
        """ Construct epsilon and mu convolution matrices """
        t1 = time.process_time()

        if self.patterns:
            idx = self.pr.idx_conv_mtx
            ems = [fft.ifftshift(self.epsi_fs.swapaxes(0, 1), axes=(0, 1)),
                   fft.ifftshift(self.epsi_inv_fs.swapaxes(0, 1), axes=(0, 1)),
                   fft.ifftshift(self.mu_fs.swapaxes(0, 1), axes=(0, 1)),
                   fft.ifftshift(self.mu_inv_fs.swapaxes(0, 1), axes=(0, 1))]
            self.epxxcm, self.epxycm, self.epyxcm, self.epyycm, self.epzzcm, \
            self.eixxcm, self.eixycm, self.eiyxcm, self.eiyycm, self.eizzcm, \
            self.muxxcm, self.muxycm, self.muyxcm, self.muyycm, self.muzzcm, \
            self.mixxcm, self.mixycm, self.miyxcm, self.miyycm, self.mizzcm, \
                = [em[idx[:, :, 0], idx[:, :, 1], i, j]
                   for em in ems
                   for i, j in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]]
        else:
            # uniform layer
            mtr = self.materials[self.material_bg]
            epsi_bg, epsi_bg_inv, mu_bg, mu_bg_inv = [mtr.epsi, mtr.epsi_inv, mtr.mu, mtr.mu_inv]  # complex 3x3 tensors
            # d = np.eye(self.pr.num_g)
            self.epxxcm, self.epxycm, self.epyxcm, self.epyycm, self.epzzcm, \
            self.eixxcm, self.eixycm, self.eiyxcm, self.eiyycm, self.eizzcm, \
            self.muxxcm, self.muxycm, self.muyxcm, self.muyycm, self.muzzcm, \
            self.mixxcm, self.mixycm, self.miyxcm, self.miyycm, self.mizzcm, \
                = [np.diag(np.full(self.pr.num_g, em[i, j], dtype=complex)) for em in [epsi_bg, epsi_bg_inv, mu_bg, mu_bg_inv] for i, j in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]]

        if self.pr.show_calc_time:
            print("{:.6f}   _cons_ep_mu_cm_3d".format(time.process_time() - t1) + ", layer "+self.name)

    def reconstruct(self,
                    n1: int = None,
                    n2: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        n1      :   number of points in the first lattice vector direction
        n2      :   number of points in the second lattice vector direction

        Returns
        -------
        xx          :   ndarray
                        x coordinates in a unit cell (2d ndarray), shape (n2, n1)
        yy          :   ndarray
                        y coordinates in a unit cell (2d ndarray), shape (n2, n1)
        epsi_recons :   ndarray
                        reconstructed epsi in a unit cell, shape (n2, n1, 3, 3)
        mu_recons   :   ndarray
                        reconstructed mu in a unit cell, shape (n2, n1, 3, 3)

        Notes
        -----
        Unit cell can be slanted. Generally xx and yy defines the grid points. can use matplotlib.pyplot to show the epsilon and mu profile, such as

            plt.pcolormesh(xx, yy, epsi_recons[:, :, 0, 0]

        to show the xx component of epsilon in a unit cell.
        """
        if n1 is None:
            n1 = 101
        if n2 is None:
            n2 = 101

        # epsi_fs and mu_fs are calculated at solving stage.
        # if call reconstruct before solving, epsi_fs and mu_fs are None.
        # Hence need to calculate them here.
        if self.pr.ks_ep_mu is None:
            self.pr._calc_ks_ep_mu()
        if (self.epsi_fs is None) or (self.mu_fs is None):
            self._calc_ep_mu_fs_3d()
        self.epsi_fs_used, self.epsi_inv_fs_used, self.mu_fs_used, self.mu_inv_fs_used = \
            [[em[j, i, :, :] for i, j in self.pr.idx_g_ep_mu_used]
             for em in [fft.ifftshift(self.epsi_fs, axes=(0, 1)), fft.ifftshift(self.epsi_inv_fs, axes=(0, 1)), fft.ifftshift(self.mu_fs, axes=(0, 1)), fft.ifftshift(self.mu_inv_fs, axes=(0, 1))]]

        eu = self.epsi_fs_used
        ea = np.array(eu, dtype=complex)
        mu = self.mu_fs_used
        ma = np.array(mu, dtype=complex)
        idx = self.pr.idx_g_ep_mu_used
        idxa = np.array(idx)

        # spatial coordinates in a unit cell of any shape
        x1 = np.linspace(-0.5, 0.5, n1)
        x2 = np.linspace(-0.5, 0.5, n2)
        xx1, xx2 = np.meshgrid(x1, x2)
        a1, a2 = [np.array(lv) for lv in self.pr.latt_vec]
        xx = xx1 * a1[0] + xx2 * a2[0]
        yy = xx1 * a1[1] + xx2 * a2[1]

        exp_term = np.exp(1j * 2 * np.pi *
                          (idxa[:, 0][:, None, None] * xx1[None, :, :] +
                           idxa[:, 1][:, None, None] * xx2[None, :, :]))
        epsi_recons = np.sum(ea[:, None, None, :, :] * exp_term[:, :, :, None, None], axis=0)
        mu_recons = np.sum(ma[:, None, None, :, :] * exp_term[:, :, :, None, None], axis=0)
        return xx, yy, epsi_recons, mu_recons

    def _calc_PQ_3d(self,
                    o: Union[float, complex] = None):
        """Calculate the P and Q matrix"""

        t1 = time.process_time()

        if o is None:
            o = self.pr.omega

        epxx, epxy, epyx, epyy, epzz, \
        eixx, eixy, eiyx, eiyy, eizz, \
        muxx, muxy, muyx, muyy, muzz, \
        mixx, mixy, miyx, miyy, mizz, \
            = [self.epxxcm, self.epxycm, self.epyxcm, self.epyycm, self.epzzcm,
               self.eixxcm, self.eixycm, self.eiyxcm, self.eiyycm, self.eizzcm,
               self.muxxcm, self.muxycm, self.muyxcm, self.muyycm, self.muzzcm,
               self.mixxcm, self.mixycm, self.miyxcm, self.miyycm, self.mizzcm]
        Kx = self.pr.Kx
        Ky = self.pr.Ky

        P11 = o * muyx + 1. / o * Kx[:, None] * eizz * Ky
        P12 = o * muyy - 1. / o * Kx[:, None] * eizz * Kx
        P21 = -o * muxx + 1. / o * Ky[:, None] * eizz * Ky
        P22 = -o * muxy - 1. / o * Ky[:, None] * eizz * Kx

        Q11 = o * epyx + 1. / o * Kx[:, None] * mizz * Ky
        Q12 = o * epyy - 1. / o * Kx[:, None] * mizz * Kx
        Q21 = -o * epxx + 1. / o * Ky[:, None] * mizz * Ky
        Q22 = -o * epxy - 1. / o * Ky[:, None] * mizz * Kx

        P = np.block([[P11, P12],
                      [P21, P22]])
        Q = np.block([[Q11, Q12],
                      [Q21, Q22]])

        if self.pr.show_calc_time:
            print("{:.6f}   _calc_PQ_3d".format(time.process_time() - t1) + ", layer "+self.name)

        return P, Q

    def _calc_eig_3d(self):
        """ Calculate the eigen modes in the layer """

        t1 = time.process_time()

        # ql2, self.phil = la.eig(- self.P @ self.Q)
        w2, v = la.eig(- self.P @ self.Q)  # w2 shape (2num_g),  v shape (2num_g, 2num_g)
        self._rad_cha = np.where(w2.real > 0)[0].tolist()  # todo: even for radiation channel, if omega.imag larger than omega.real, q02.real is negative
        w = np.sqrt(w2 + 0j)

        w = self._w_sign_channel(w, w2)

        wis0 = (np.abs(w) == 0.)  # array[True or False], if w is 0
        wn0 = np.logical_not(wis0)
        i_wis0 = np.where(wis0)
        i_wn0 = np.where(wn0)

        vh = np.zeros((2 * self.pr.num_g, 2 * self.pr.num_g), dtype=complex)
        vh[:, i_wn0[0]] = -1j * (self.Q @ v[:, i_wn0[0]]) / w[i_wn0[0]]
        # vh = -1j * q @ v / w[:, None, :]

        if wis0.any():
            w2h_, vh_ = la.eig(- self.Q @ self.P)

            o = self.pr.omega
            _o = o * (1 + 1e-13)
            P, Q = self._calc_PQ_3d(_o)

            _w2, _v = la.eig(-P @ Q)  # w2 shape (num_g, 2), v shape (num_g, 2, 2)
            _w2 = _w2  # shape (num_g, 2)
            _w = np.sqrt(_w2 + 0j)
            _v_w0 = _v[:, i_wis0[0]]
            _vh = -1j * Q @ _v_w0 / _w[i_wis0[0]]

            for ii in range(len(i_wis0[0])):
                _vh_norm = np.sqrt(np.conj(_vh[ii]) @ _vh[ii])
                _v_norm = np.sqrt(np.conj(_v_w0[ii]) @ _v_w0[ii])
                if _vh_norm >= _v_norm:
                    # vh[i_wis0[0][ii], :, i_wis0[1][ii]] /= _vh_norm
                    vh[:, i_wis0[0][ii]] = vh_[:, i_wis0[0][ii]]
                    v[:, i_wis0[0][ii]] = 0.
                else:
                    # the column of v is already correct
                    vh[:, i_wis0[0][ii]] = np.array([[0.],
                                                     [0.]])

        # normalize such that the larger norm of v and vh's each column is 1
        vn = sla.norm(v, axis=0)
        vhn = sla.norm(vh, axis=0)
        nm = np.maximum(vn, vhn)
        v /= nm
        vh /= nm

        phil = v
        psil = vh


        # # debugging,  check if phi is eigen and consistent with psi
        # ng = self.pr.num_g
        # P = self.P
        # Q = self.Q
        #
        # psil1 = -1j * Q @ phil / w
        # diff = np.abs(psil1 - psil).max()
        # print('psi0 diff {:g}'.format(diff))
        # diff_where = np.where(np.abs(psil1 - psil) > 1e-10)
        #
        # check_eigen = P @ Q @ phil
        # diff1 = np.abs(check_eigen + phil * w * w).max()
        # print('check eigen {:g}'.format(diff1))
        # # a = 1

        # # old
        # ql_inv = 1. / self.ql
        # self.psil = -1j * self.Q @ self.phil * ql_inv
        # # self.psil = 1j * sla.solve(self.P, self.phil) @ np.diag(self.ql)

        self.ql = w
        self.phil = phil
        self.psil = psil

        if self.pr.show_calc_time:
            print("{:.6f}   _calc_eig_3d".format(time.process_time() - t1) + ", layer "+self.name)

    def _calc_pq_3d_uniform(self, o, mxx, mxy, myx, myy, mzz, exx, exy, eyx, eyy, ezz, kxa, kya):
        """
        calculate pq of 3d uniform layer
        Parameters
        ----------
        o
        mxx
        mxy
        myx
        myy
        mzz
        exx
        exy
        eyx
        eyy
        ezz
        kxa
        kya

        Returns
        -------

        """
        p = np.array([[o * myx + 1. / o / ezz * kxa * kya, o * myy - 1. / o / ezz * kxa ** 2],
                      [-o * mxx + 1. / o / ezz * kya ** 2, -o * mxy - 1. / o / ezz * kxa * kya]], dtype=complex)  # (2, 2, num_g) shape
        q = np.array([[o * eyx + 1. / o / mzz * kxa * kya, o * eyy - 1. / o / mzz * kxa ** 2],
                      [-o * exx + 1. / o / mzz * kya ** 2, -o * exy - 1. / o / mzz * kxa * kya]], dtype=complex)  # (2, 2, num_g) shape
        p = np.rollaxis(p, -1)  # shape (num_g, 2, 2)
        q = np.rollaxis(q, -1)  # shape (num_g, 2, 2)
        pq = p @ q  # shape (num_g, 2, 2)

        qp = q @ p  # shape (num_g, 2, 2)

        return p, q, pq, qp

    def _w_sign_channel(self, w, w2):
        """
        w is supposed to be generated by
        `w = np.sqrt(w2 + 0j)`

        Parameters
        ----------
        w2

        Returns
        -------

        """
        # w = np.sqrt(w2 + 0j)

        w = w.copy()

        if self.in_mid_out == 'mid':
            w[w.imag < 0] *= -1
        else:
            if self.pr.omega.imag < 0:
                if self.pr.ccnif == "physical":
                    w[(w2.real < 0) * (w.imag < 0)] *= -1
                elif self.pr.ccnif == "ac":
                    w[(w2.real < 0) * (w.imag > 0)] *= -1
                else:
                    warn("ccnif not recognized. Default to 'physical'.")
                    w[(w2.real < 0) * (w.imag < 0)] *= -1
            elif self.pr.omega.imag > 0:
                if self.pr.ccpif == "ac":
                    w[(w2.real < 0) * (w.imag < 0)] *= -1
                elif self.pr.ccpif == "physical":
                    w[(w2.real < 0) * (w.imag > 0)] *= -1
                else:
                    warn("ccpif not recognized. Default to 'ac'.")
                    w[(w2.real < 0) * (w.imag < 0)] *= -1
            else:
                w[w.imag < 0] *= -1

        return w

    def _calc_eig_3d_uniform(self):
        """ Efficient calculation of eigen for uniform layer """

        t1 = time.process_time()

        if self.is_vac:
            self.ql = self.pr.q0
            self.phil = self.pr.phi0
            self.psil = self.pr.psi0
            self._rad_cha = self.pr.rad_cha_0 + [i+self.pr.num_g for i in self.pr.rad_cha_0]
            self.phil_2x2s = self.pr.phi0_2x2s

        else:  # not vacuum
            # before fixing Wood: for numG 100 this takes about 2ms

            kxa, kya = [a.ravel() for a in np.hsplit(np.array(self.pr.ks), 2)]
            o = self.pr.omega
            Kx = self.pr.Kx.copy()
            Ky = self.pr.Ky.copy()
            mtr = self.materials[self.material_bg]
            mxx, mxy, myx, myy, mzz, exx, exy, eyx, eyy, ezz = [a[i, j] for a in [mtr.mu, mtr.epsi] for i, j in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]]

            p, q, pq, qp = self._calc_pq_3d_uniform(o, mxx, mxy, myx, myy, mzz, exx, exy, eyx, eyy, ezz, kxa, kya)

            # if (np.abs(mxy) + np.abs(myx) + np.abs(exy) + np.abs(eyx)) == 0. and (eyy/ezz == myy/mzz) and (exx/ezz == mxx/mzz):
            if self.is_dege:
            # if False:  # this is for debugging
                # direct construction of eigen, no solving

                # # Using identity as phil
                # # self._phil_is_idt = True
                # ng = self.pr.num_g
                # phil = np.eye(2*self.pr.num_g, dtype=complex)
                # v = np.eye(2, dtype=complex)[None, :, :]  # for later use in constructing psi
                # w2 = - (pq[:, range(2), range(2)])  # shape (num_g, 2)
                # w = np.sqrt(w2 + 0j)
                # w = self._w_sign_channel(w, w2)
                # row = np.array([[0, 0], [ng, ng]])
                # rows = np.repeat(row[:, :, None], ng, axis=2)
                # column = np.array([[0, ng], [0, ng]])
                # columns = np.repeat(column[:, :, None], ng, axis=2)
                # self.phil_2x2s = phil[rows, columns]
                # ql = w.T.ravel()  # 1d array length 2num_g
                # vh = -1j * q @ v / w[:, None, :]
                # psil = np.zeros((2*ng, 2*ng), dtype=complex)
                # r1 = range(ng)
                # r2 = range(ng, 2 * ng)
                # psil[r1, r1] = vh[:, 0, 0]
                # psil[r1, r2] = vh[:, 0, 1]
                # psil[r2, r1] = vh[:, 1, 0]
                # psil[r2, r2] = vh[:, 1, 1]


                # construct phil and psil that is Wood-stable
                w2 = o**2 * exx * myy - myy/mzz * Ky * Ky - exx / ezz * Kx * Kx
                w2 = np.concatenate([w2, w2])
                self._rad_cha = np.where(w2.real > 0)[0].tolist()  # todo: even for radiation channel, if omega.imag larger than omega.real, q02.real is negative
                w = np.sqrt(w2 + 0j)

                w = self._w_sign_channel(w, w2)

                ql = w.T.ravel()  # 1d array length 2num_g

                o = self.pr.omega
                Kx = self.pr.Kx.copy()
                Ky = self.pr.Ky.copy()
                k_norm = np.sqrt(np.conj(Kx) * Kx + np.conj(Ky) * Ky)
                alpha = ezz / mzz

                ng = self.pr.num_g
                qlh = ql[:ng]  # 1d array of length num_g
                # skc = 0.05
                # i_knz = np.where(k_norm < (skc * np.abs(o)))[0]
                i_kez = np.where(k_norm == 0.)[0]  # k is zero
                # i_qsw = np.where((np.abs(qlh) <= np.abs(o)) * (k_norm >= (skc * np.abs(o))))[0]
                i_qsw = np.where((np.abs(qlh) < np.abs(o)))[0]
                i_qlw = np.where(np.abs(qlh) > np.abs(o))[0]
                idxa = np.array(self.pr.idx_g)
                ii = (idxa[:, 0] == 0) & (idxa[:, 1] == 0)

                c1 = np.array([eyy * Ky, -exx * Kx], dtype=complex)
                c2 = np.array([Kx, Ky], dtype=complex)
                c1f = np.ones(ng, dtype=complex)
                c2f = c1f.copy()

                # c1[:, i_knz] = np.array([[1.], [0.]])
                # c2[:, i_knz] = -1j / o / alpha * np.array([1. / mzz * Kx[i_knz] * Ky[i_knz] / qlh[i_knz], 1./myy*(-exx/ezz*np.square(Kx[i_knz]) - np.square(qlh[i_knz])) / qlh[i_knz]])  # should not be |Kx|^2
                c1[:, i_kez] = np.array([[1.], [0.]], dtype=complex)
                c2[:, i_kez] = np.array([[0.], [1.]], dtype=complex)
                cphi = np.cos(self.pr._phi)
                sphi = np.sin(self.pr._phi)
                c1[:, ii] = np.array([[eyy * sphi], [-exx * cphi]], dtype=complex)
                # c1[:, ii] = np.array([[sphi], [-cphi]], dtype=complex)
                c2[:, ii] = np.array([[cphi], [sphi]], dtype=complex)

                c1f[i_qlw] = o / qlh[i_qlw] / k_norm[i_qlw]
                c2f[i_qlw] = 1j / k_norm[i_qlw]

                c1f[i_qsw] = 1. / k_norm[i_qsw]
                c2f[i_qsw] = 1j / o * qlh[i_qsw] / k_norm[i_qsw]

                # c1f[i_knz] = 1.
                # c2f[i_knz] = 1.

                c1f[i_kez] = 1.
                c2f[i_kez] = 1j * qlh[i_kez] / o

                c1f[ii] = 1.
                c2f[ii] = 1j * qlh[ii] / o

                c1 *= c1f
                c2 *= c2f

                r1 = range(ng)
                r2 = range(ng, 2 * ng)
                phil = np.zeros((2*ng, 2*ng), dtype=complex)
                psil = phil.copy()
                phil[r1, r1] = c1[0, :]
                phil[r2, r1] = c1[1, :]
                phil[r1, r2] = c2[0, :]
                phil[r2, r2] = c2[1, :]
                psil[r1, r1] = c2[0, :] * alpha
                psil[r2, r1] = c2[1, :] * alpha
                psil[r1, r2] = c1[0, :]
                psil[r2, r2] = c1[1, :]

                self.phil_2x2s = np.moveaxis(np.array([c1, c2]), 0, 1)


                # # debugging, check if phi is eigen and consistent with psi
                # P = np.zeros((2 * ng, 2 * ng), dtype=complex)
                # r1 = range(ng)
                # r2 = range(ng, 2 * ng)
                # P[r1, r1] = p[:, 0, 0]
                # P[r2, r1] = p[:, 1, 0]
                # P[r1, r2] = p[:, 0, 1]
                # P[r2, r2] = p[:, 1, 1]
                # Q = ezz/mzz * P
                #
                # psil1 = -1j * Q @ phil / ql
                # diff = np.abs(psil1 - psil).max()
                # print('psi0 diff {:g}'.format(diff))
                # diff_where = np.where(np.abs(psil1 - psil)>1e-10)
                #
                # check_eigen = P @ Q @ phil
                # diff1 = np.abs(check_eigen + phil * ql * ql).max()
                # print('check eigen {:g}'.format(diff1))
                # pass

            else:  # require solving of 2x2 PQ Hamiltonian
                w2, v = la.eig(-pq)  # w2 shape (num_g, 2), v shape (num_g, 2, 2)
                w2 = w2  # shape (num_g, 2)
                self._rad_cha = np.where(w2.real > 0)[0].tolist()  # todo: even for radiation channel, if omega.imag larger than omega.real, q02.real is negative
                w = np.sqrt(w2 + 0j)

                w = self._w_sign_channel(w, w2)

                ql = w.T.ravel()  # 1d array length 2num_g

                wis0 = (np.abs(w) == 0.)  # array[True or False], if w is 0
                wn0 = np.logical_not(wis0)
                i_wis0 = np.where(wis0)
                i_wn0 = np.where(wn0)

                vh = np.zeros((self.pr.num_g, 2, 2), dtype=complex)
                vh[i_wn0[0], :, i_wn0[1]] = -1j * (q[i_wn0[0], :, :] @ v[i_wn0[0], :, i_wn0[1]][:, :, None])[:, :, 0] / w[i_wn0[0], i_wn0[1], None]

                if wis0.any():
                    w2h_, vh_ = la.eig(-qp)

                    _o = o * (1 + 1e-13)
                    _p, _q, _pq, _qp = self._calc_pq_3d_uniform(_o, mxx, mxy, myx, myy, mzz, exx, exy, eyx, eyy, ezz, kxa, kya)
                    _w2, _v = la.eig(-_pq)  # w2 shape (num_g, 2), v shape (num_g, 2, 2)
                    _w2 = _w2  # shape (num_g, 2)
                    _w = np.sqrt(_w2 + 0j)
                    _v_w0 = _v[i_wis0[0], :, i_wis0[1]]
                    _vh = -1j * (_q[i_wis0[0], :, :] @ _v_w0[:, :, None])[:, :, 0] / _w[i_wis0[0], i_wis0[1], None]

                    for ii in range(len(i_wis0[0])):
                        _vh_norm = np.sqrt(np.conj(_vh[ii]) @ _vh[ii])
                        _v_norm = np.sqrt(np.conj(_v_w0[ii]) @ _v_w0[ii])
                        if _vh_norm >= _v_norm:
                            # vh[i_wis0[0][ii], :, i_wis0[1][ii]] /= _vh_norm
                            vh[i_wis0[0][ii], :, i_wis0[1][ii]] = vh_[i_wis0[0][ii], :, i_wis0[1][ii]]
                            v[i_wis0[0][ii], :, i_wis0[1][ii]] = np.array([0., 0.])
                        else:
                            # the column of v is already correct
                            vh[i_wis0[0][ii], :, i_wis0[1][ii]] = np.array([0., 0.])

                vn = sla.norm(np.moveaxis(v, 1, 2).reshape(self.pr.num_g*2, 2), axis=1).reshape(self.pr.num_g, 2)[:, None, :]
                vhn = sla.norm(np.moveaxis(vh, 1, 2).reshape(self.pr.num_g * 2, 2), axis=1).reshape(self.pr.num_g, 2)[:, None, :]
                nm = np.maximum(vn, vhn)
                v /= nm
                vh /= nm

                ng = self.pr.num_g
                r1 = range(ng)
                r2 = range(ng, 2 * ng)

                phil = np.zeros((2*ng, 2*ng), dtype=complex)
                phil[r1, r1] = v[:, 0, 0]
                phil[r1, r2] = v[:, 0, 1]
                phil[r2, r1] = v[:, 1, 0]
                phil[r2, r2] = v[:, 1, 1]

                psil = np.zeros((2*ng, 2*ng), dtype=complex)
                psil[r1, r1] = vh[:, 0, 0]
                psil[r2, r1] = vh[:, 1, 0]
                psil[r1, r2] = vh[:, 0, 1]
                psil[r2, r2] = vh[:, 1, 1]

                row = np.array([[0, 0], [ng, ng]])
                rows = np.repeat(row[:, :, None], ng, axis=2)
                column = np.array([[0, ng], [0, ng]])
                columns = np.repeat(column[:, :, None], ng, axis=2)
                self.phil_2x2s = phil[rows, columns]

                # # debugging, check if phil is eigen and consistent with psil
                # ng = self.pr.num_g
                # P = np.zeros((2 * ng, 2 * ng), dtype=complex)
                # r1 = range(ng)
                # r2 = range(ng, 2 * ng)
                # P[r1, r1] = p[:, 0, 0]
                # P[r2, r1] = p[:, 1, 0]
                # P[r1, r2] = p[:, 0, 1]
                # P[r2, r2] = p[:, 1, 1]
                # # Q = ezz/mzz * P
                #
                # Q = np.zeros((2 * ng, 2 * ng), dtype=complex)
                # r1 = range(ng)
                # r2 = range(ng, 2 * ng)
                # Q[r1, r1] = q[:, 0, 0]
                # Q[r2, r1] = q[:, 1, 0]
                # Q[r1, r2] = q[:, 0, 1]
                # Q[r2, r2] = q[:, 1, 1]
                #
                # psil1 = -1j * Q @ phil / ql
                # diff = np.abs(psil1 - psil).max()
                # print('psi0 diff {:g}'.format(diff))
                # diff_where = np.where(np.abs(psil1 - psil)>1e-10)
                #
                # check_eigen = P @ Q @ phil
                # diff1 = np.abs(check_eigen + phil * ql * ql).max()
                # print('check eigen {:g}'.format(diff1))
                # a = 1

            # ql = w.T.ravel()  # 1d array length 2num_g
            # # vh = -1j * q @ v / w[:, None, :]
            # vh = -1j * p @ v / w[:, None, :]
            #
            # ng = self.pr.num_g
            # r1 = range(ng)
            # r2 = range(ng, 2 * ng)
            # psil = np.zeros((2*ng, 2*ng), dtype=complex)
            # psil[r1, r1] = vh[:, 0, 0]
            # psil[r2, r1] = vh[:, 1, 0]
            # psil[r1, r2] = vh[:, 0, 1]
            # psil[r2, r2] = vh[:, 1, 1]
            # # psil = np.block([[psil11, psil12],
            # #                  [psil21, psil22]])

            self.ql = ql
            self.phil = phil
            self.psil = psil

        if self.pr.show_calc_time:
            print("{:.6f}   _calc_eig_3d_uniform".format(time.process_time() - t1) + ", layer "+self.name)

    def _calc_PQ_2d(self,
                    o: Optional[Union[float, complex]] = None):

        if o is None:
            o = self.pr.omega
        eixx = self.eixxcm
        muyy = self.muyycm
        eizz = self.eizzcm
        mixx = self.mixxcm
        epyy = self.epyycm
        mizz = self.mizzcm
        Kx = self.pr.Kx
        Ky = self.pr.Ky

        # TM is p, Ex-Hy-Ez
        Ppilu = sla.lu_factor(-1./o*eixx)  # i means inverted
        Qp = o * muyy - 1. / o * Kx[:, None] * eizz * Kx

        # TE is s, Hx-Ey-Hz
        Psilu = sla.lu_factor(-1./o * mixx)
        Qs = o * epyy - 1. / o * Kx[:, None] * mizz * Kx

        return Ppilu, Qp, Psilu, Qs

    def _calc_eig_2d(self):
        """
        Calculate the eigen modes in purely 2D scenario, i.e. when TE and TM are separated.

        Notes
        -----
        Three conditions need to be met for this subroutine to be applicable
        *   structure is 2D, uniform in y
        *   no off-diagonal components in permittivity and permeability
        *   incident waves in x-z plane, no phi rotation.
        """
        t1 = time.process_time()

        # o = self.pr.omega
        # eixx = self.eixxcm
        # muyy = self.muyycm
        # eizz = self.eizzcm
        # mixx = self.mixxcm
        # epyy = self.epyycm
        # mizz = self.mizzcm
        # Kx = self.pr.Kx
        # Ky = self.pr.Ky
        #
        # # TM is p, Ex-Hy-Ez
        # Ppilu = sla.lu_factor(-1./o*eixx)  # i means inverted
        # Qp = o * muyy - 1. / o * Kx[:, None] * eizz * Kx
        #
        # # TE is s, Hx-Ey-Hz
        # Psilu = sla.lu_factor(-1./o * mixx)
        # Qs = o * epyy - 1. / o * Kx[:, None] * mizz * Kx

        Ppilu, Qp, Psilu, Qs = self._calc_PQ_2d()

        _phil = []
        _psil = []
        _ql = []
        _rc = []  # radiation channel

        for P, Q in [(Ppilu, Qp), (Psilu, Qs)]:

            # ql2, phil = la.eig(- sla.lu_solve(P, Q))
            # rc = np.where(ql2.real > 0)[0].tolist()  # todo: even for radiation channel, if omega.imag larger than omega.real, q02.real is negative
            # ql = np.sqrt(ql2 + 0j)
            #
            # ql = self._w_sign_channel(ql, ql2)
            #
            # ql_inv = 1. / ql
            # psil = -1j * Q @ phil * ql_inv
            # # psil = 1j * la.inv(P) @ phil @ np.diag(self.ql)

            w2, v = la.eig(- sla.lu_solve(P, Q))
            rc = np.where(w2.real > 0)[0].tolist()  # todo: even for radiation channel, if omega.imag larger than omega.real, q02.real is negative
            w = np.sqrt(w2 + 0j)

            w = self._w_sign_channel(w, w2)

            wis0 = (np.abs(w) == 0.)  # array[True or False]. if w is 0
            wn0 = np.logical_not(wis0)
            i_wis0 = np.where(wis0)
            i_wn0 = np.where(wn0)

            # for non zero w, calculate vh using -jQv/q
            vh = np.zeros((self.pr.num_g, self.pr.num_g), dtype=complex)
            vh[:, i_wn0[0]] = -1j * (Q @ v[:, i_wn0[0]]) / w[i_wn0[0]]
            # where w is 0, vh's column is 0

            # normalize such that the larger norm of v and vh's each column is 1
            vn = sla.norm(v, axis=0)
            vhn = sla.norm(vh, axis=0)
            nm = np.maximum(vn, vhn)
            v /= nm
            vh /= nm

            phil = v
            psil = vh

            # # debugging,  check if phi is eigen and consistent with psi
            # ng = self.pr.num_g
            #
            # psil1 = -1j * Q @ phil / w
            # diff = np.abs(psil1 - psil).max()
            # print('psi0 diff {:g}'.format(diff))
            # diff_where = np.where(np.abs(psil1 - psil) > 1e-10)
            #
            # check_eigen = sla.lu_solve(P, Q) @ phil
            # diff1 = np.abs(check_eigen + phil * w * w).max()
            # print('check eigen {:g}'.format(diff1))
            # # a = 1

            _phil.append(phil)
            _psil.append(psil)
            _ql.append(w)
            _rc.append(rc)

        self.phil = np.zeros((2*self.pr.num_g, 2*self.pr.num_g), dtype=complex)
        self.phil[:self.pr.num_g, :self.pr.num_g] = _psil[0]
        self.phil[self.pr.num_g:, self.pr.num_g:] = _phil[1]

        self.psil = np.zeros((2*self.pr.num_g, 2*self.pr.num_g), dtype=complex)
        self.psil[:self.pr.num_g, self.pr.num_g:] = _psil[1]
        self.psil[self.pr.num_g:, :self.pr.num_g] = _phil[0]

        self.ql = np.concatenate(_ql)
        self._rad_cha = _rc[0] + [a+self.pr.num_g for a in _rc[1]]

        if self.pr.show_calc_time:
            print("{:.6f}   _calc_eig_2d".format(time.process_time() - t1) + ", layer "+self.name)

    def _calc_im(self):
        """Calculate the interface matrix."""

        t1 = time.process_time()

        # # old calc of iml0, layer sandwiched in vac
        # if self.is_vac:
        #     self.iml0 = self.pr.im0
        # else:
        #     al0, bl0 = im(self.phil, self.psil, self.pr.phi0, self.pr.psi0, self._phil_is_idt)
        #     self.iml0 = (al0, bl0)
        #
        # # calc of im0l, for debugging, assume layers sandwiched in vac
        # a0l, b0l = im(self.pr.phi0, self.pr.psi0, self.phil, self.psil)
        # self.imfl = (a0l, b0l)

        # iml0 (actually imlf),
        # al0, bl0 = im(self.phil, self.psil, self.pr.phif, self.pr.psif)
        # self.iml0 = (al0, bl0)

        # calc imfl
        phif = self.pr.phif
        psif = self.pr.psif
        phil = self.phil
        psil = self.psil
        # term1 = phif @ phil
        term1 = phil  # attention! check if phif is eye
        term2 = psif @ psil
        a0l = term1 + term2
        b0l = term1 - term2
        self.imfl = (a0l, b0l)


        # # debugging

        # # term1 = self.pr.phif2 @ self.phil
        # # term2 = self.pr.psif2 @ self.psil
        # # a0l2 = term1 + term2
        # # b0l2 = term1 - term2
        # a0l2, b0l2 = im(self.pr.phi0, self.pr.psi0, self.phil, self.psil)
        # self.imfl2 = (a0l2, b0l2)

        # term1 = self.pr.phif3 @ self.phil
        # term2 = self.pr.psif3 @ self.psil
        # a0l3 = term1 + term2
        # b0l3 = term1 - term2
        # self.imfl3 = (a0l3, b0l3)

        # term1 = sla.solve(self.pr.phif4, self.phil)
        # term2 = sla.solve(self.pr.psif4, self.psil)
        # a0l4 = term1 + term2
        # b0l4 = term1 - term2
        # self.imfl4 = (a0l4, b0l4)

        # # end of debugging

        if self.pr.show_calc_time:
            print("{:.6f}   _calc_im".format(time.process_time() - t1) + ", layer "+self.name)

    def _calc_sm(self):
        """ calculate the scattering matrix of current layer """

        t1 = time.process_time()

        # if self.is_vac and self.thickness == 0:
        #     self.sm = self.pr.sm0
        # else:
        if self.in_mid_out == 'mid':
            if self.thickness == 0:
                sm = self.pr.sm0
            else:
                # sm = s_1l(self.thickness, self.ql, *self.iml0)

                # try:
                sm = s_1l_rsp(self.thickness, self.ql, *self.imfl)
                # except Exception:
                #     print('layer name:' + self.name)
                #     print('omega: {:g}'.format(self.pr.omega))
                #     print('theta: {:g}'.format(self.pr.theta))
                #     print('ql:')
                #     print(self.ql)
                #     print(self.pr.idx_g)
                #     print(self.pr.ks)

                # sm = s_1l_rsp_lrd(self.thickness, self.ql, *self.imfl, *self.imfl4)
        elif self.in_mid_out == 'in':
            # sm = s_1l_1212(*self.iml0)
            sm = s_1l_1221(*self.imfl)
        elif self.in_mid_out == 'out':
            # sm = s_1l_1221(*self.iml0)
            sm = s_1l_1212(*self.imfl)
        else:
            raise Exception('Layer is not the incident, a middle, or the output layer.')
        self.sm = sm

        if self.pr.show_calc_time:
            print("{:.6f}   _calc_sm layer".format(time.process_time() - t1) + ", layer "+self.name)

    def solve(self):
        """
        calculate the eigen, interface matrix, and scattering matrix of the layer in vacuum
        Returns
        -------

        """
        t1 = time.process_time()

        if self.if_mod:
            self._calc_ep_mu_fs_3d()
            self._cons_ep_mu_cm_3d()
            if self.patterns:
                if self.pr.is_1d_latt and self.pr.phi == 0. and not self.materials_ode:
                    self._calc_eig_2d()
                else:
                    P, Q = self._calc_PQ_3d()
                    self.P = P
                    self.Q = Q
                    self._calc_eig_3d()
            else:
                self._calc_eig_3d_uniform()

                # # for debugging
                # P, Q = self._calc_PQ_3d()
                # self.P = P
                # self.Q = Q
                # self._calc_eig_3d()

                # # for debugging
                # self._calc_eig_2d()

            self._calc_im()
        if self.if_mod or self._if_t_change:
            self._calc_sm()
            self.if_mod = False
            self._if_t_change = False

        if self.pr.show_calc_time:
            print('{:.6f}'.format(time.process_time() - t1) + '   layer ' + self.name + ' solve total')

