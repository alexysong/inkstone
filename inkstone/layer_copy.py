# -*- coding: utf-8 -*-

from typing import Tuple, Optional, List
import time
import numpy as np
from inkstone.layer import Layer
from inkstone.sm import s_1l, s_1l_in, s_1l_out


class LayerCopy:

    def __init__(self, name: str, layer: Layer, thickness: float):
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
        self._al_bl: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.in_mid_out: str = 'mid'  # {'in', 'mid', 'out'}, if this layer is the incident, output, or a middle layer

        self.sm: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._if_t_change = True
        self.if_mod = True  # simulator is responsible for triggering if_mod for all
        self.need_recalc_al_bl = True

        self.thickness = thickness

    def __getattr__(self, item):
        """ignore all calls if not explicitly defined"""
        # raise Exception('what the hell')
        # print(1+1)
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
    def im(self):
        return self.layer.im

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
    def al_bl(self) -> Optional[Tuple[np.ndarray, np.ndarray]] :
        return self._al_bl

    @al_bl.setter
    def al_bl(self, val: Tuple[np.ndarray, np.ndarray]):
        self._al_bl = val

    @property
    def is_vac(self) -> bool:
        return self.layer.is_vac

    @property
    def rad_cha(self) -> List[int]:
        return self.layer.rad_cha

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

        if self.is_vac and self.thickness == 0:
            self.sm = self.layer.pr.sm0
        else:
            if self.in_mid_out == 'mid':
                if self.thickness == 0:
                    sm = self.layer.pr.sm0
                else:
                    sm = s_1l(self.thickness, self.layer.ql, *self.layer.im)
            elif self.in_mid_out == 'in':
                sm = s_1l_in(*self.layer.im)
            elif self.in_mid_out == 'out':
                sm = s_1l_out(*self.layer.im)
            else:
                raise Exception('Layer is not the incident, a middle, or the output layer.')
            self.sm = sm

        if self.layer.pr.show_calc_time:
            print('{:.6f}   _calc_sm  (layer copy)'.format(time.process_time() - t1))

