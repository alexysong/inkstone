# -*- coding: utf-8 -*-

from typing import Tuple, Optional, List
import time
from inkstone.layer import Layer
from inkstone.sm import s_1l, s_1l_1212, s_1l_1221, s_1l_rsp
from warnings import warn
from GenericBackend import genericBackend as gb


class LayerCopy:

    def __init__(self, name: str, layer: Layer, thickness: float,gb=gb):
        """
        A dummy layer for layer copies

        Parameters
        ----------
        name        :   the name of this layer
        layer       :   the original layer
        thickness   :   thickness of this layer
        """

        self.name = name
        self.is_copy = True
        self.layer = layer
        self.original_layer_name = layer.name
        self.pr = self.layer.pr
        self._al_bl: Optional[Tuple[any, any]] = None
        self._in_mid_out: str = 'mid'  # {'in', 'mid', 'out'}, if this layer is the incident, output, or a middle layer

        self.sm: Optional[Tuple[any, any]] = None
        self._if_t_change = True
        self.if_mod = True  # simulator is responsible for triggering if_mod for all
        self.need_recalc_al_bl = True

        self._thickness = thickness
        self.thickness = thickness

    def __getattr__(self, item):
        """ignore all calls if not explicitly defined"""
        # raise Exception('what the hell')
        # print(1+1)
        warn('You might have called something that is not defined in a copy layer: "{:s}". Instead you should call it from the original layer: "{:s}". Nothing is changed.'.format(self.name, self.original_layer_name))
        pass

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, val):
        if self._thickness != val:
            self._thickness = val
            self._if_t_change = True
            self.if_mod = True

    @property
    def materials_used(self):
        return self.layer.materials_used

    # @property
    # def if_mod(self):
    #     return self.layer.if_mod
    #
    # @if_mod.setter
    # def if_mod(self, val):
    #     pass

    @property
    def iml0(self):
        return self.layer.iml0

    @property
    def imfl(self):
        return self.layer.imfl

    @property
    def ql(self):
        return self.layer.ql

    @property
    def phil(self):
        return self.layer.phil

    @property
    def psil(self):
        return self.layer.psil

    @property
    def eizzcm(self):
        return self.layer.eizzcm

    @property
    def mizzcm(self):
        return self.layer.mizzcm

    @property
    def al_bl(self) -> Optional[Tuple[any, any]] :
        return self._al_bl

    @al_bl.setter
    def al_bl(self, val: Tuple[any, any]):
        self._al_bl = val

    @property
    def is_vac(self) -> bool:
        return self.layer.is_vac

    @property
    def is_isotropic(self) -> bool:
        return self.layer.is_isotropic

    @property
    def is_diagonal(self) -> bool:
        return self.layer.is_diagonal

    @property
    def is_dege(self) -> bool:
        return self.layer.is_dege

    @property
    def rad_cha(self) -> List[int]:
        return self.layer.rad_cha

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
            warn('You are setting the background material of a layer that is itself a copy. As a result, the original layer and all of its copies including this one will be changed.')
            self.layer.set_layer(material_bg=material_bg)

    def add_box(self,
                mtr: str,
                shp: str,
                box_name: str = None,
                **kwargs):
        warn('You are adding some patterns to a copy layer: "{:s}". Instead, this pattern is added to the original layer: "{:s}" and all of its copies.'.format(self.name, self.original_layer_name))
        self.layer.add_box(mtr=mtr, shp=shp, box_name=box_name, **kwargs)

    def set_box(self,
                box_name: str,
                **kwargs):
        warn('You are updating some patterns to a copy layer: "{:s}". Instead, this pattern is updated in the original layer: "{:s}" and all of its copies.'.format(self.name, self.original_layer_name))
        self.layer.set_box(box_name=box_name, **kwargs)


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
                    self.pr.ind_inci = gb.sqrt(mbg.epsi[0, 0] * mbg.mu[0, 0])
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
                    self.pr.ind_out = gb.sqrt(mbg.epsi[0, 0] * mbg.mu[0, 0])
            else:
                self.pr.out_is_vac = False
                self.pr.out_is_iso_nonvac = False
                self.pr.ind_out = None


    # def _set_pr_inci_out(self):
    #     self.layer._set_pr_inci_out()

    def reconstruct(self,
                    n1: int = None,
                    n2: int = None) -> Tuple[any, any, any, any]:
        warn('You are requesting the reconstruction of a copy layer: "{:s}". The reconstruction of the original layer "{:s}" is returned.'.format(self.name, self.original_layer_name))
        return self.layer.reconstruct(n1=n1, n2=n2)

    def solve(self):
        t1 = time.process_time()
        if self.layer.if_mod or self.if_mod or self._if_t_change:
            self.layer.solve()
            self._calc_sm()
            self._if_t_change = False
            self.if_mod = False
        if self.layer.pr.show_calc_time:
            print('{:.6f}'.format(time.process_time() - t1) + '   layer ' + self.name+' solve (layer copy)')

    def _calc_sm(self):
        """ calculate the scattering matrix of current layer """

        t1 = time.process_time()

        # if self.is_vac and self.thickness == 0:
        #     self.sm = self.layer.pr.sm0
        # else:
        if self.in_mid_out == 'mid':
            if self.thickness == 0:
                sm = self.layer.pr.sm0
            else:
                # sm = s_1l(self.thickness, self.layer.ql, *self.layer.iml0)
                sm = s_1l_rsp(self.thickness, self.ql, *self.imfl)
        elif self.in_mid_out == 'in':
            # sm = s_1l_1212(*self.iml0)
            sm = s_1l_1221(*self.layer.imfl)
        elif self.in_mid_out == 'out':
            # sm = s_1l_1221(*self.iml0)
            sm = s_1l_1212(*self.layer.imfl)
        else:
            raise Exception('Layer is not the incident, a middle, or the output layer.')
        self.sm = sm

        if self.layer.pr.show_calc_time:
            print('{:.6f}   _calc_sm  (layer copy)'.format(time.process_time() - t1))

