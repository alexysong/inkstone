# -*- coding: utf-8 -*-
# import scipy.sparse as sps
import scipy.linalg as sla
from warnings import warn
from collections import OrderedDict
from typing import Tuple, Optional, List, Dict, Union
import time
from GenericBackend import genericBackend as gb

from inkstone.rsp import rsp, rsp_sa12lu, rsp_sb21lu
from inkstone.params import Params
from inkstone.mtr import Mtr
from inkstone.layer import Layer
from inkstone.layer_copy import LayerCopy


class Inkstone:
    # todo: more tests of magneto-optics and gyro-magnetic

    def __init__(self,
                 lattice: Optional[Union[float, Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
                 num_g: int = None,
                 omega: Union[float, complex] = None,
                 frequency: Union[float, complex] = None,
                 theta: float = None,
                 phi: float = None,
                 gb = gb
                 ):
        """

        Main routine.

        Parameters
        ----------
        lattice         :   in-plane lattice vector(s)
        omega           :   angular frequency, 2pi c / wavelength
        frequency       :   c/wavelength. since I use c=1, frequency is really 1/wavelength
        num_g           :   number of G point
        theta, phi      :   incident angle in degrees
        """
        # global parameters
        self.pr: Params = Params(latt_vec=lattice, num_g=num_g,
                                 omega=omega, frequency=frequency, theta=theta, phi=phi,
                                 )

        self._need_recalc_sm: bool = True  # if the structure has been modified

        self._layers_mod: List[int] = []  # the indices of the layers that are modified.

        # this is not needed
        # self._inci_changed: bool = True  # if incidence changed

        self.ai: Optional[any] = None  # incident wave amplitudes from the incident region.
        self.bo: Optional[any] = None  # incident wave amplitudes from the output region
        self.ao: Optional[any] = None  # transmitted wave amplitudes in the output region
        self.bi: Optional[any] = None  # reflected wave amplitudes in the incident region
        self._need_recalc_bi_ao: bool = True

        # thickness of all layers, and cumulative thickness
        self.thicknesses: OrderedDict[str, float] = OrderedDict()
        self.total_thickness: float = 0.
        self.thicknesses_c: List[float] = []

        self.materials: Dict[str, Mtr] = {'vacuum': Mtr(1, 1, name='vacuum')}  # materials used

        self.layers: OrderedDict[str, Layer] = OrderedDict()  # all layers

        self.sm: Optional[Tuple[any, any, any, any]] = None
        self.csms: List[List[Optional[Tuple[int, int, Tuple[any, any, any, any]]]]] = []  # the cumulative scattering matrices.
        self.csmsr: List[Optional[Tuple[int, int, Tuple[any, any, any, any]]]] = []  # the cumulative scattering matrices reversed.
        
        self.gb = gb

    @property
    def lattice(self) -> Union[float, Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        The lattice vectors.

        For 2D calculations (1D in-plane), all the patterns should have shape of "1d".
        """
        return self.pr.latt_vec
    lattice.__doc__ += Params.latt_vec.__doc__
    __init__.__doc__ += lattice.__doc__

    @lattice.setter
    def lattice(self, val):
        if val != self.pr.latt_vec:
            self.pr.latt_vec = val
            for layer_name, layer in self.layers.items():
                layer.if_mod = True

    @property
    def num_g(self) -> int:
        """"""
        return self.pr.num_g
    num_g.__doc__ += Params.num_g.__doc__

    @num_g.setter
    def num_g(self, val):
        if self.pr.num_g != val:
            self.pr.num_g = val
            for layer_name, layer in self.layers.items():
                layer.if_mod = True

    @property
    def omega(self) -> Union[float, complex]:
        """"""
        return self.pr.omega
    omega.__doc__ += Params.omega.__doc__

    @omega.setter
    def omega(self, val):
        self.pr.omega = val
        if self.pr.omega != val:
            self.pr.omega = val
            for layer_name, layer in self.layers.items():
                layer.if_mod = True

    @property
    def frequency(self) -> Union[float, complex]:
        """"""
        return self.pr.frequency
    frequency.__doc__ += Params.frequency.__doc__

    @frequency.setter
    def frequency(self, val):
        # self.pr.frequency = val
        if self.pr.frequency != val:
            self.pr.frequency = val
            for layer_name, layer in self.layers.items():
                layer.if_mod = True

    @property
    def theta(self):
        """"""
        return self.pr.theta
    theta.__doc__ += Params.theta.__doc__

    @theta.setter
    def theta(self, val):
        if (val is not None) and (val != self.pr.theta):
            if val is not self.gb.raw_type:
                val = self.gb.parseData(val)
            self.pr.theta = val
            for layer_name, layer in self.layers.items():
                layer.if_mod = True

    @property
    def phi(self):
        """"""
        return self.pr.phi
    phi.__doc__ += Params.phi.__doc__

    @phi.setter
    def phi(self, val):
        if (val is not None) and (val != self.pr.phi):
            self.pr.phi = val
            for layer_name, layer in self.layers.items():
                layer.if_mod = True

    def SetLattice(self,
                   lattice_vectors: Tuple[Tuple[float, float], Tuple[float, float]]):
        """
        Set the lattice vectors。
        """
        self.lattice = lattice_vectors
    SetLattice.__doc__ += lattice.__doc__

    def SetNumG(self, num_g: int):
        """
        Set the number of G points.
        """
        self.num_g = num_g
    SetNumG.__doc__ += num_g.__doc__

    def channels_choices(self,
                         n: str = None,
                         p: str = None
                         ):
        """
        Parameters
        ----------
        n    :   {"Physical", "ac}, default: "physical"
        p    :   {"Physical", "ac}, default: "ac"

        Returns
        -------

        """
        if p is not None:
            self.pr.ccpif = p
        if n is not None:
            self.pr.ccnif = n

    def AddMaterial(self,
                    name: str,
                    epsilon: Union[Union[float, complex], Tuple[Union[float, complex], Union[float, complex], Union[float, complex]], any],
                    mu: Union[Union[float, complex], Tuple[Union[float, complex], Union[float, complex], Union[float, complex]], any] = 1.
                    ):
        """
        Add materials to structure.

        A material is defined by its permittivity tensor epsilon and permeability tensor mu. Both epsilon and mu can be a number, a tuple of 3, or (3, 3) array.

        Notes
        -----
        `vacuum` is a built-in material, can directly use it. Use this pre-defined `vacuum` will give faster calculations than user re-defined `vacuum`.

        """
        if name in ['vacuum', 'Vacuum']:
            warn('Material "vacuum" is built-in. This command is not executed.', UserWarning)
        else:
            self.materials[name] = Mtr(epsilon, mu, name=name)

    def SetMaterial(self,
                    name: str,
                    epsi: Union[Union[float, complex], Tuple[Union[float, complex], Union[float, complex], Union[float, complex]], any] = None,
                    mu: Union[Union[float, complex], Tuple[Union[float, complex], Union[float, complex], Union[float, complex]], any] = None
                    ):
        """
        Update material parameters.
        """
        mtr = self.materials[name]
        if epsi is not None:
            mtr.epsi = epsi
        if mu is not None:
            mtr.mu = mu
        if (epsi is not None) or (mu is not None):
            for layer_name, layer in self.layers.items():
                if name in layer.materials_used:
                    layer.if_mod = True

    def AddLayer(self,
                 name: str,
                 thickness: float,
                 material_background: str):
        """
        Add a layer to the structure.

        Parameters
        ----------
        name                :   name of the layer
        thickness           :   regardless of user input, the first layer and the last layer's thicknesses are set to 0
        material_background :   background material
        """
        if name not in self.layers.keys():
            layer = Layer(name, thickness, material_background, self.materials, self.pr)
            if not self.layers:
                if thickness != 0.:
                    warn('You set the first layer (incident region) thickness to be nonzero. This thickness is ignored and set to 0, i.e. treated as infinity. If you meant there was an infinite vacuum before this layer, please explicitly add that using AddLayer().')
                    thickness = 0.
                layer.in_mid_out = 'in'
            self.layers[name] = layer
            self.thicknesses[name] = thickness
            self.total_thickness += thickness
            self.thicknesses_c.append(self.total_thickness)
            self.csms.append([])
            # self._determine_layers()
        else:
            warn('A layer with the given name already exists. This new layer is NOT added.')

    def AddLayerCopy(self,
                     name: str,
                     original_layer: str,
                     thickness: float):
        """
        Add a layer copy.

        Its thickness can be different from the original.

        Parameters
        ----------
        name            :   name of this layer
        original_layer  :   name of the layer being copied
        thickness       :   thickness of the layer copy. Can be different than original.

        """
        if name not in self.layers.keys():
            layer = self.layers[original_layer]
            layer_copy = LayerCopy(name, layer, thickness)
            self.layers[name] = layer_copy

            self.thicknesses[name] = thickness
            self.total_thickness += thickness
            self.thicknesses_c.append(self.total_thickness)

            self.csms.append([])
            # self._determine_layers()
        else:
            warn('A layer with the given name already exists. This new layer is NOT added.')

    def SetLayer(self,
                 name: str,
                 thickness: float = None,
                 material_bg: str = None):
        """
        Reset layer parameters

        Parameters
        ----------
        name        :   the layer to modify
        thickness   :   change the layer thickness
        material_bg :   choose a different background material.

        """
        if name in self.layers.keys():
            layer = self.layers[name]
            if thickness is not None and thickness != layer.thickness:
                if layer.in_mid_out == 'in':
                    warn("The incident region thickness needs to be 0 and can't be changed.", RuntimeWarning)
                    thickness = 0.
                elif layer.in_mid_out == 'out':
                    warn("The output region thickness needs to be 0 and can't be changed.", RuntimeWarning)
                    thickness = 0.
                layer.set_layer(thickness=thickness)
                self.thicknesses[name] = thickness
                self._calc_thicknesses()
            if material_bg is not None and material_bg != layer.material_bg:
                layer.set_layer(material_bg=material_bg)

                # Updates to inci layer parameters in `Params` is called through layer.set_layer - material_bg setter - _set_pr_inci_out()

                # propagate if_mod of the layer to all layer copies
                if layer.if_mod:
                    for ly in self.layers.values():
                        if ly.original_layer_name == layer.original_layer_name:
                            ly.if_mod = True
            # self._determine_layers()
        else:
            warn('Did not find the layer you specified. The layer is NOT changed.', UserWarning)

    def ReconstructLayer(self,
                         name: str,
                         nx: int = None,
                         ny: int = None) -> Tuple[any, any, any, any]:
        """
        Reconstruct layer permittivity and permeability profile

        Returns xx, yy, epsilon, mu. xx and yy are the results of `self.gb.meshgrid`. The shape of epsilon and mu are (ny, nx, 3, 3), where the last two axis is the tensor at the spatial locations.

        Use `matplotlib.pyplot.pcolormesh` to plot the reconstruction. For example to plot the (0, 0) element of the permittivity profile,

            plt.pcolormesh(xx, yy, epsilon[:, :, 0, 0])

        Parameters
        ----------
        name    :   name of the layer
        nx      :   number of points in x direction
        ny      :   number of points in y direction

        Returns
        -------
        xx          :
                        x coordinates in a unit cell (2d ndarray), shape (n2, n1)
        yy          :
                        y coordinates in a unit cell (2d ndarray), shape (n2, n1)
        epsi        :
                        reconstructed epsi in a unit cell, shape (n2, n1, 3, 3)
        mu          :
                        reconstructed mu in a unit cell, shape (n2, n1, 3, 3)

        """
        if name in self.layers.keys():
            result = self.layers[name].reconstruct(nx, ny)
            return result
        else:
            warn('Did not find the layer you specified.', UserWarning)

    def _calc_thicknesses(self):
        """
        calculate total thickness and cumulative thickness
        """
        self.total_thickness = 0.
        self.thicknesses_c = []
        for layer_name, thickness in self.thicknesses.items():
            self.total_thickness += thickness
            self.thicknesses_c.append(self.total_thickness)

    def AddPattern(self,
                   layer: str,
                   material: str,
                   shape: str,
                   pattern_name: Optional[str] = None,
                   **kwargs
                   ):
        """
        Add a pattern to a layer.

        The shapes should not intersect with one another, but may be contained in one another or share a side.

        Parameters
        ----------
        layer           :   the name of the layer to add the pattern
        material        :   name of the material in this pattern.
        shape           :   {'rectangle', 'disk', 'ellipse', 'polygon', '1d'}
                            for 2d calculations (1d in-plane), all patterns should be in '1d' shape.
        pattern_name    :   the name of the pattern.
                            In one layer, each pattern name should be unique.
                            If not given, its name would be automatically set, for example as "box3" if it is the 4th box added to this layer.
        kwargs          :   other arguments to set up the pattern.
                            to be passed on to different shapes in `~.shps.Rect`, `~.shps.Disk`, `~.shps.Elli`, `~.shps.Poly`, `~.shp.OneD`.
                            can also contain arguments to control Gibbs correction for that shape, to be passed on to `~.gibbs.gibbs_corr`.

        Examples
        --------
            s.AddPattern(name="slab", material="vacuum", shape="rectangle", pattern_name="rect1",
                         side_lengths=(0.5, 0.4), center=(0., 0.), angle=0., **kw_gibbs)

            s.AddPattern(name="slab", material="vacuum", shape="parallelogram", pattern_name="para1",
                         side_lengths=(0.5, 0.4), center=(0., 0.), angle=0., shear_angle=80., **kw_gibbs)

            s.AddPattern(name="slab", material="vacuum", shape="disk", pattern_name="disk1",
                         radius=0.2, center=(0., 0.), **kw_gibbs)

            s.AddPattern(name="slab", material="vacuum", shape="ellipse", pattern_name="elli1",
                         half_lengths=(0.3, 0.2), center=(0.1, 0.05), angle=30., **kw_gibbs)

            s.AddPattern(name="slab", material="vacuum", shape="polygon", pattern_name="poly1",
                         vertices=[(0, 0), (0.4, 0), (0.5, 0.4), (0.1, 0.5)], **kw_gibbs)

            s.AddPattern(name="slab", material="vacuum", shape="1d", pattern_name="box",
                         width=0.4, center=0.1, **kw_gibbs)
        """

        if self.pr.is_1d_latt and shape != '1d':
            warn('This is a 2D calculation (i.e. 1D in-plane). Setting 2D in-plane patterns may lead to unexpected results.', RuntimeWarning)
        if not self.pr.is_1d_latt and shape == '1d':
            warn('This is a 3D calculation (i.e. 2D in-plane). Setting 1D in-plane patterns may lead to unexpected results.', RuntimeWarning)

        if layer in self.layers.keys():
            self.layers[layer].add_box(material, shape, box_name=pattern_name, **kwargs)
            # propagate if_mod of the layer to all layer copies
            if self.layers[layer].if_mod:
                for ly in self.layers.values():
                    if ly.original_layer_name == self.layers[layer].original_layer_name:
                        ly.if_mod = True
        else:
            warn('Did not find the layer you specified. The layer is not changed.', UserWarning)

    def AddPattern1D(self,
                     layer: str,
                     material: str,
                     width: float,
                     center: float = None,
                     pattern_name: Optional[str] = None,
                     **kw_gibbs
                     ):
        if not self.pr.is_1d_latt:
            warn('This is a 3D calculation (i.e. 2D in-plane). Setting 1D in-plane patterns may lead to unexpected results.', RuntimeWarning)

        if layer in self.layers.keys():
            self.layers[layer].add_box(material, "1d", box_name=pattern_name, width=width, center=center, **kw_gibbs)
            # propagate if_mod of the layer to all layer copies
            if self.layers[layer].if_mod:
                for ly in self.layers.values():
                    if ly.original_layer_name == self.layers[layer].original_layer_name:
                        ly.if_mod = True
        else:
            warn('Did not find the layer you specified. The layer is not changed.', UserWarning)

    def AddPatternRectangle(self,
                            layer: str,
                            material: str,
                            side_lengths: Tuple[float, float],
                            center: Tuple[float, float] = None,
                            angle: float = None,
                            pattern_name: Optional[str] = None,
                            **kw_gibbs
                            ):
        if self.pr.is_1d_latt:
            warn('This is a 2D calculation (i.e. 1D in-plane). Setting 2D in-plane patterns may lead to unexpected results.', RuntimeWarning)

        if layer in self.layers.keys():
            self.layers[layer].add_box(material, "rectangle", box_name=pattern_name, side_lengths=side_lengths, center=center, angle=angle, **kw_gibbs)
            # propagate if_mod of the layer to all layer copies
            if self.layers[layer].if_mod:
                for ly in self.layers.values():
                    if ly.original_layer_name == self.layers[layer].original_layer_name:
                        ly.if_mod = True
        else:
            warn('Did not find the layer you specified. The layer is not changed.', UserWarning)

    def AddPatternParallelogram(self,
                                layer: str,
                                material: str,
                                side_lengths: Tuple[float, float],
                                center: Tuple[float, float] = None,
                                angle: float = None,
                                shear_angle: float = None,
                                pattern_name: Optional[str] = None,
                                **kw_gibbs
                                ):

        if self.pr.is_1d_latt:
            warn('This is a 2D calculation (i.e. 1D in-plane). Setting 2D in-plane patterns may lead to unexpected results.', RuntimeWarning)

        if layer in self.layers.keys():
            self.layers[layer].add_box(material, "parallelogram", box_name=pattern_name, side_lengths=side_lengths, center=center, angle=angle, shear_angle=shear_angle, **kw_gibbs)
            # propagate if_mod of the layer to all layer copies
            if self.layers[layer].if_mod:
                for ly in self.layers.values():
                    if ly.original_layer_name == self.layers[layer].original_layer_name:
                        ly.if_mod = True
        else:
            warn('Did not find the layer you specified. The layer is not changed.', UserWarning)

    def AddPatternDisk(self,
                       layer: str,
                       material: str,
                       radius: float,
                       center: Tuple[float, float] = None,
                       pattern_name: Optional[str] = None,
                       **kw_gibbs
                       ):
        if self.pr.is_1d_latt:
            warn('This is a 2D calculation (i.e. 1D in-plane). Setting 2D in-plane patterns may lead to unexpected results.', RuntimeWarning)

        if layer in self.layers.keys():
            self.layers[layer].add_box(material, "disk", box_name=pattern_name, radius=radius, center=center, **kw_gibbs)
            # propagate if_mod of the layer to all layer copies
            if self.layers[layer].if_mod:
                for ly in self.layers.values():
                    if ly.original_layer_name == self.layers[layer].original_layer_name:
                        ly.if_mod = True
        else:
            warn('Did not find the layer you specified. The layer is not changed.', UserWarning)

    def AddPatternEllipse(self,
                          layer: str,
                          material: str,
                          half_lengths: Tuple[float, float],
                          center: Tuple[float, float] = None,
                          angle: float = None,
                          pattern_name: Optional[str] = None,
                          **kw_gibbs
                          ):
        if self.pr.is_1d_latt:
            warn('This is a 2D calculation (i.e. 1D in-plane). Setting 2D in-plane patterns may lead to unexpected results.', RuntimeWarning)

        if layer in self.layers.keys():
            self.layers[layer].add_box(material, "ellipse", box_name=pattern_name, half_lengths=half_lengths, center=center, angle=angle, **kw_gibbs)
            # propagate if_mod of the layer to all layer copies
            if self.layers[layer].if_mod:
                for ly in self.layers.values():
                    if ly.original_layer_name == self.layers[layer].original_layer_name:
                        ly.if_mod = True
        else:
            warn('Did not find the layer you specified. The layer is not changed.', UserWarning)

    def AddPatternPolygon(self,
                          layer: str,
                          material: str,
                          vertices: List[Tuple[float, float]],
                          pattern_name: Optional[str] = None,
                          **kw_gibbs
                          ):
        if self.pr.is_1d_latt:
            warn('This is a 2D calculation (i.e. 1D in-plane). Setting 2D in-plane patterns may lead to unexpected results.', RuntimeWarning)

        if layer in self.layers.keys():
            self.layers[layer].add_box(material, "polygon", box_name=pattern_name, vertices=vertices, **kw_gibbs)
            # propagate if_mod of the layer to all layer copies
            if self.layers[layer].if_mod:
                for ly in self.layers.values():
                    if ly.original_layer_name == self.layers[layer].original_layer_name:
                        ly.if_mod = True
        else:
            warn('Did not find the layer you specified. The layer is not changed.', UserWarning)

    def SetPattern(self, layer_name, pattern_name: str, **kwargs):
        if layer_name in self.layers.keys():
            self.layers[layer_name].set_box(pattern_name, **kwargs)
            # propagate if_mod of the layer to all layer copies
            layer = self.layers[layer_name]
            if layer.if_mod:
                for ly in self.layers.values():
                    if ly.original_layer_name == layer.original_layer_name:
                        ly.if_mod = True
        else:
            warn('Did not find the layer you specified. The layer is NOT changed.', UserWarning)

    def SetExcitation(self,
                      theta: Union[float, complex] = None,
                      phi: float = None,
                      s_amplitude: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                      p_amplitude: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                      order: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
                      s_amplitude_back: Union[float, List[float]] = None,
                      p_amplitude_back: Union[float, List[float]] = None,
                      order_back: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
                      kx: Union[float, complex] = None,
                      ky: Union[float, complex] = None,
                      ):
        """
        Set the excitation plane wave.

        Parameters
        ----------
        theta               :
        phi                 :
                                incident angles in units of degrees.
                                `theta` is the oblique angle from normal (z).
                                `phi` is the azimuthal angle, the angle from x axis to the in-plane projection of the incident k, ccw means positive.
                                Note, these are angles in the incident region (the angles in the output region could be different and are automatically calculated).
        s_amplitude         :   Electric field amplitude of incident s wave.
                                "s" means electric field parallel to xy plane, perpendicular to incident plane. The incident plane contains z and k.
                                A list would mean the electric field amplitude of s-wave incidence in several orders, defined by `order`
        p_amplitude         :   "p" means electric field polarized in the incident plane.
                                See `s_amplitude` definition.
        order               :   The Fourier order(s) of incident waves
        s_amplitude_back    :   backside incident s wave from the output region
        p_amplitude_back    :   backside incident p wave form the output region
        order_back          :   Fourier order for the backside incidence
        kx, ky              :   the in-plane wave vector of the incident waves.


        Notes
        -----
        the s and p amplitudes and the orders in the front side and the backside can all be lists.

        """

        li: Layer = list(self.layers.values())[0]
        lo: Layer = list(self.layers.values())[-1]

        if ((theta is not None) or (phi is not None)) and ((kx is not None) or (ky is not None)):
            raise Exception("Please specify either (theta, phi) or (kx, ky) but not at the same time.")

        if theta is not None:
            if phi is not None:
                self.pr.iesbtpsp = True
                self.pr.iesbksp = False
                self.pr.iesbe = False
                self.theta = theta
                self.phi = phi
            else:
                raise Exception("Both theta and phi need to be set.")
        elif phi is not None:
                raise Exception("Both theta and phi need to be set.")

        if kx is not None:
            if ky is not None:
                self.pr.iesbksp = True
                self.pr.iesbtpsp = False
                self.pr.iesbe = False
                self.pr.k_pa_inci = (kx, ky)
                for layer_name, layer in self.layers.items():
                    layer.if_mod = True
            else:
                raise Exception("Both kx and ky need to be set.")
        elif ky is not None:
            raise Exception("Both kx and ky need to be set.")

        if (s_amplitude is not None) or (p_amplitude is not None) or (order is not None) or (s_amplitude_back is not None) or (p_amplitude_back is not None) or (order_back is not None):
            self.pr.set_inci_ord_amp(s_amplitude, p_amplitude, order, s_amplitude_back, p_amplitude_back, order_back)
            self._need_recalc_bi_ao = True
            for ly in list(self.layers.values()):
                ly.need_recalc_al_bl = True

        if (not li.is_isotropic) and (s_amplitude or p_amplitude):
            raise Exception('Incident region is not uniform isotropic. s and p waves with arbitrary angles are not guaranteed to be eign. Use `SetExcitationByEigen()` instead.')

        if (not lo.is_isotropic) and (s_amplitude_back or p_amplitude_back):
            raise Exception('Output region is not uniform isotropic. s and p waves with arbitrary angles are not guaranteed to be eign. If you are setting incidence from this back side, please consider use `SetExcitationByEigen()` instead.')


    def SetExcitationByEigen(self,
                             kx: Union[float, complex],
                             ky: Union[float, complex],
                             a: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                             # sa: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                             # pa: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                             eigen_number: Union[int, List[int]] = None,
                             ab: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                             # sab: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                             # pab: Optional[Union[float, complex, List[Union[float, complex]]]] = None,
                             eigen_number_back:  Union[int, List[int]] = None
                             ):
        """
        Set excitation to the structure by specifying in-plane wave vectors kx and ky and the amplitudes of the eigen waves in the incident and output regions.

        This method can be useful when the incident and the output regions are not isotropic or not uniform.

        Parameters
        ----------
        kx
        ky
        # sa
        # pa
        a :
            sa and pa are the complex electric field amplitude of the s and p waves if the incident layer supports s and p waves as eigen.
            When the incident layer's eigen are not s and p waves, sa and pa are the electric field amplitudes of the two eigen modes at this (kx, ky).
            A list would mean the electric field amplitudes of incidence in several eigen numbers, defined by `en`.
        eigen_number:
        # sab
        # pab
        ab
            sab and pab are the amplitudes of the waves incident from the backside "output" region
        eigen_number_back :

        Returns
        -------

        """
        # todo: test this API
        # Note: can't set s and p. only uniform not is_dege can have [for certain kx ky there are two eigen modes not degenerate in q]. non-uniform， one kx ky is not an eigen.
        # Note: do eigens have fixed (kx, ky)? no... They are a combination of Fourier orders.

        en = eigen_number
        enb = eigen_number_back

        if (en is None) and (enb is None):
            raise Exception('You must select at least an eigen number from either the incident side or the output side.')
        else:
            ae = []
            for (_a, _n) in [(a, en), (ab, enb)]:
                if _n is not None:
                    if not hasattr(_n, "__len__"):
                        _n = [_n]
                    _n = self.gb.parseData(_n)

                    if _a is None:
                        raise Exception('You input eigen number but not its amplitude.')
                    elif not hasattr(_a, "__len__"):
                        _a = [_a]
                    _a = self.gb.parseData(_a)

                    if len(_a) != len(_n):
                        raise Exception('The length of the amplitudes and the eigen numbers are not the same.')
                else:
                    if _a is not None:
                        raise Exception('You input eigen amplitude but not the eigen number.')

                ae.append(_a)
                ae.append(_n)
        a, en, ab, enb = ae

        self.pr._s_amps = None
        self.pr._s_amps_bk = None
        self.pr._p_amps = None
        self.pr._p_amps_bk = None
        self.pr._theta = None
        self.pr._phi = None
        # self.pr.sin_phis, self.pr.cos_phis, self.pr.sin_varthetas, self.pr.cos_varthetas are auto updated when setting inci kx and ky. No need to reset them

        self.pr.k_pa_inci = (kx, ky)

        ll = list(self.layers.values())
        li = ll[0]
        # lo = ll[-1]
        # o = self.pr.omega
        if li.is_isotropic:
            warn('Uniform isotropic medium. You can use `SetExcitation()` which works with theta, phi, s and p wave amplitudes.', UserWarning)
            # Note: should not reinvent the wheel. Could just call SetExcitation() from here, but then possible confusion in self.pr.iesbe and self.pr.iesbtpsp.
            kn = self.gb.sqrt(self.gb.abs(kx) ** 2 + self.gb.abs(ky) ** 2)
            if kn != 0.:
                self.pr._phi = self.gb.arccos(kx / kn)  # not setting property self.pr.phi to avoid double calling stuff like self.pr._calc_ks()
            else:
                warn("At normal incidence, phi is uncertain and default to 0.")
                self.pr._phi = 0.
            self.pr._theta = self.gb.arcsin(kn / self.pr.kii.real)
            # problem: in this case, not sure which eigen is s and p, when user sets ai, user doesn't know the eigen which is not solved yet.
            # logically, whether or not isotropic, should delete theta phi. However, uniform layer's eigen choice depend on theta phi (1.  s and p happen to give Wood stability 2 eigen choosing s and p is user friendly, and ai bo simple).
            # if kx=ky=0, ambiguity in phi still. In the code, the internally chosen eigen decides the effective phi. It could even be the two degenerate eigen are chosen to be not orthonormal, hence phi not defined. But then again as of [202310] The eigens in uniform isotropic is chosen by s and p using cos(phi) and sin(phi)
        else:
            self.pr._theta = None
            self.pr._phi = None

        # todo: lo, bo?

        aibo = []
        for z, n in zip([a, ab], [en, enb]):
            i = self.gb.zeros(2*self.pr.num_g, dtype=self.gb.complex128)
            i[n] = z
            aibo.append(i)
            # todo: is this done?

        self.ai, self.bo = aibo

        self.pr.iesbe = True
        self.pr.iesbtpsp = False
        self.pr.iesbksp = False


    def SetFrequency(self, freq: Union[float, complex]):
        """
        set the frequency

        Parameters
        ----------
        freq   :   incident wave frequency.
        """
        self.frequency = freq

    def _calc_ai_bo_3d(self):
        """calculate incident ai and bo amplitudes"""

        # t1 = time.process_time()

        o = self.pr.omega

        layer_inci: Layer = list(self.layers.values())[0]
        aibo = []
        if self.pr.iesbe:
            pass

        elif self.pr.iesbtpsp or self.pr.iesbksp:

            for ii, (sa, pa, od, sphi, cphi, sthe, cthe) in enumerate([[self.pr._s_amps, self.pr._p_amps, self.pr._incident_orders, self.pr.sin_phis, self.pr.cos_phis, self.pr.sin_varthetas, self.pr.cos_varthetas], [self.pr._s_amps_bk, self.pr._p_amps_bk, self.pr._incident_orders_bk, self.pr.sin_phis_bk, self.pr.cos_phis_bk, self.pr.sin_varthetas_bk, self.pr.cos_varthetas_bk]]):
                if self.pr._num_g_ac:
                    ab = self.gb.zeros(2 * self.pr._num_g_ac) + 0j
                    if (sa or pa) and od and self.pr.idx_g and \
                            sphi and cphi and sthe and cthe:
                        # find the index of the input orders in the g list
                        idx = [i for order in od for i, j in enumerate(self.pr.idx_g) if j == order]
                        for i, jj in enumerate(idx):
                        # for i in range(len(idx)):
                            if (jj in self.pr.q0_0):
                                # todo: need to handle this and document it.
                                # if user specify 90 degree incidence, this is activated
                                warn('You are specifying incidence in a channel that is parallel to the surface of the structure. \n In this case, only specific field configuration is allowed.')
                                ab[jj] = sa[i]
                                ab[jj + self.pr._num_g_ac] = pa[i]
                            else:
                                s = sa[i]
                                p = pa[i]
                                sp = sphi[jj]
                                cp = cphi[jj]
                                st = sthe[jj]
                                ct = cthe[jj]

                                # # original
                                # ab[i] = -s * self.sin_phis[i] + p * self.sin_varthetas[i] * self.cos_phis[i]  # e_x
                                # ab[i + self._num_g_ac] = s * self.cos_phis[i] + p * self.sin_varthetas[i] * self.sin_phis[i]  # e_y

                                # # original, corrected
                                # ab[jj] = -s * self.sin_phis[jj] + p * self.sin_varthetas[jj] * self.cos_phis[jj]  # e_x
                                # ab[jj + self._num_g_ac] = s * self.cos_phis[jj] + p * self.sin_varthetas[jj] * self.sin_phis[jj]  # e_y

                                # with new calc that removed convergence problem at Wood
                                ex = -s * sp + p * st * cp  # e_x
                                ey = s * cp + p * st * sp  # e_y
                                phi_2x2 = self.gb.castType(layer_inci.phil_2x2s[:, :, jj],self.gb.complex128)
                                v = self.gb.la.solve(phi_2x2, [ex, ey])
                                ab[jj] = v[0]
                                ab[jj + self.pr._num_g_ac] = v[1]

                    aibo.append(ab)
        else:
            warn("Haven't set excitation yet.", RuntimeWarning)

        self.ai, self.bo = aibo
        # print('_calc_ai_bo_3d', time.process_time() - t1)

    def _calc_al_bl_layer(self, i: int):
        """Calculate the field coefficients al and bl of a layer"""

        n_layers = len(self.layers)
        layersl = list(self.layers.values())
        layer = layersl[i]

        if layer.need_recalc_al_bl:

            t1 = time.process_time()

            if layer.in_mid_out == 'in':
                layer.al_bl = (self.ai, self.bi)

            elif layer.in_mid_out == 'out':
                layer.al_bl = (self.ao, self.bo)
            else:

                self._calc_csmr_layer(i)
                self._calc_csmr_layer(i + 1)
                csmr = layersl[i].csmr
                csmrn = layersl[i + 1].csmr

                self._calc_csm_layer(i)
                self._calc_csm_layer(i - 1)
                csm = layersl[i].csm
                csmp = layersl[i-1].csm

                # # # # # # # # # #
                # # old method using al0, bl0.
                # # works with either vac or fic gap layers. With new method using a0l and b0l and fic material, csm and csmr all different because inci/output vac's sm is different
                # # reshape 1d array ai and bo into a 2d 1-column vector
                # ai_v = self.pr.ai.reshape((2 * self.pr.num_g, 1))
                # bo_v = self.pr.bo.reshape((2 * self.pr.num_g, 1))
                #
                # al0 = layer.iml0[0]
                # bl0 = layer.iml0[1]
                # I = self.gb.diag(self.gb.ones(2 * self.pr.num_g, dtype=self.gb.complex128))  # identity
                # # csm
                # sc11, sc12, sc21, sc22 = csm
                # # csm of previous layer
                # scp11, scp12, scp21, scp22 = csmp
                # # csmr
                # sci11, sci12, sci21, sci22 = csmr
                # # csmr of next layer
                # scin11, scin12, scin21, scin22 = csmrn
                #
                # if (layersl[i-1].in_mid_out == "in") and not layersl[i-1].is_vac:
                #     sa = sla.lu_solve(scp21, ai_v)
                #     sb = sci12 @ bo_v
                #     al = 1. / 2. * (bl0 @ sla.solve((I - sci11 @ scp22), (sci11 @ sa + sb))
                #                     + al0 @ sla.solve((I - scp22 @ sci11), (sa + scp22 @ sb)))
                # else:
                #     sa = scp21 @ ai_v
                #     sb = sci12 @ bo_v
                #     al = 1. / 2. * (bl0 @ sla.solve((I - sci11 @ scp22), (sci11 @ sa + sb))
                #                     + al0 @ sla.solve((I - scp22 @ sci11), (sa + scp22 @ sb)))
                #
                # if (layersl[i+1].in_mid_out == "out") and not layersl[i+1].is_vac:
                #     sa = sc21 @ ai_v
                #     sb = sla.lu_solve(scin12, bo_v)
                #     bl = 1. / 2. * (bl0 @ sla.solve((I - sc22 @ scin11), (sa + sc22 @ sb))
                #                     + al0 @ sla.solve((I - scin11 @ sc22), (scin11 @ sa + sb)))
                # else:
                #     sa = sc21 @ ai_v
                #     sb = scin12 @ bo_v
                #     bl = 1. / 2. * (bl0 @ sla.solve((I - sc22 @ scin11), (sa + sc22 @ sb))
                #                     + al0 @ sla.solve((I - scin11 @ sc22), (scin11 @ sa + sb)))
                #
                # # ravel 2d array (just one column) to 1d array
                # al_old = al.ravel()
                # bl_old = bl.ravel()
                # layer.al_bl = (al_old, bl_old)
                # # # # # # # # # #

                # with fic material, no use of al0, bl0, using a0l and b0l instead
                # works with either vac or fic gap layers
                I = self.gb.diag(self.gb.ones(2 * self.pr.num_g, dtype=self.gb.complex128))  # identity
                # csm of previous layer
                scp11, scp12, scp21, scp22 = csmp
                # csmr of next layer
                scin11, scin12, scin21, scin22 = csmrn

                a, b = layer.imfl
                # ia = la.inv(a)
                aTlu = sla.lu_factor(a.T)
                aTlu2 = (self.gb.clone(aTlu[0]), self.gb.clone(aTlu[1]))
                a1 = aTlu2[0]
                a1[self.gb.triu_indices(a1.shape[0])] *= 0.5

                alu = sla.lu_factor(a)
                alu2 = (alu[0].copy(), alu[1].copy())
                a1 = alu2[0]
                a1[self.gb.triu_indices(a1.shape[0])] *= 0.5

                ab = sla.lu_solve(alu, b)

                # alu = sla.lu_factor(a)
                # aib = sla.lu_solve(alu, b)
                # sl11 = b @ ia
                sl11 = sla.lu_solve(aTlu, b.T).T
                # sl12 = a - b @ ia @ b
                sl12 = 0.5 * (a - b @ ab)
                sl21 = alu2
                sl22 = -ab

                ql = layer.ql
                thickness = layer.thickness
                f = self.gb.exp(1j * ql * thickness)

                sr11 = sl22
                sr12 = sl21
                sr21 = sl12
                sr22 = sl11

                from .rsp import rsp_sb21lu, rsp_sa12lu, rsp_sa12lu_sb21lu
                if (layersl[i-1].in_mid_out == "in"):
                    scp11, scp12, scp21, scp22 = rsp_sa12lu_sb21lu(scp11, scp12, scp21, scp22, sl11, sl12, sl21, sl22)
                    # scp11, scp12, scp21, scp22 = rsp_sa12lu(scp11, scp12, scp21, scp22, sl11, sl12, sl21, sl22)  # for debugging
                else:
                    scp11, scp12, scp21, scp22 = rsp_sb21lu(scp11, scp12, scp21, scp22, sl11, sl12, sl21, sl22)

                if (layersl[i+1].in_mid_out == "out"):
                    scin11, scin12, scin21, scin22 = rsp_sa12lu_sb21lu(sr11, sr12, sr21, sr22, scin11, scin12, scin21, scin22)
                    # scin11, scin12, scin21, scin22 = rsp_sb21lu(sr11, sr12, sr21, sr22, scin11, scin12, scin21, scin22)  # for debugging
                else:
                    scin11, scin12, scin21, scin22 = rsp_sa12lu(sr11, sr12, sr21, sr22, scin11, scin12, scin21, scin22)

                sa = scp21 @ self.ai
                sb = scin12 @ self.bo
                # try:
                al = sla.solve((I - (scp22 * f) @ (scin11 * f)), (sa + (scp22 * f) @ sb))
                bl = sla.solve((I - (scin11 * f) @ (scp22 * f)), ((scin11 * f) @ sa + sb))
                # except Exception as e:
                #     warn('Singular matrix in calculating al and bl.')
                #     print(la.cond((I - (scp22 * f) @ (scin11 * f))))
                #     # print(la.cond((sa + (scp22 * f) @ sb)))
                #     print(la.cond((I - (scin11 * f) @ (scp22 * f))))
                #     # print(la.cond(((scin11 * f) @ sa + sb)))
                #     al = sla.solve((I - (scp22 * f) @ (scin11 * f) + 1e-14 * I), (sa + (scp22 * f) @ sb))
                #     bl = sla.solve((I - (scin11 * f) @ (scp22 * f) + 1e-14 * I), ((scin11 * f) @ sa + sb))
                #     print(al.max())
                #     print(bl.max())

                al = al.ravel()
                bl = bl.ravel()

                layer.al_bl = (al, bl)

            layer.need_recalc_al_bl = False

            if self.pr.show_calc_time:
                print('{:.6f}   al bl'.format(time.process_time() - t1))

    def _calc_bi_ao(self):
        """
        calculate the reflected bi and transmitted ao
        """
        # this takes a little bit of time (~1ms)

        if self._need_recalc_bi_ao:
            t1 = time.process_time()
            ai_v = self.ai.reshape((2 * self.pr.num_g, 1))
            bo_v = self.bo.reshape((2 * self.pr.num_g, 1))
            sm11 = self.sm[0]
            sm12 = self.sm[1]
            sm21 = self.sm[2]
            sm22 = self.sm[3]

            bi = sm11 @ ai_v + sm12 @ bo_v
            ao = sm21 @ ai_v + sm22 @ bo_v

            # ravel 2d array (just one column) to 1d array
            bi = bi.ravel()
            ao = ao.ravel()
            self.bi = bi
            self.ao = ao
            self._need_recalc_bi_ao = False

            if self.pr.show_calc_time:
                print('{:.6f}   bi ao'.format(time.process_time() - t1))

    def _calc_sm(self):
        """
        Calculate the scattering matrix of the entire structure.
        """
        # Dynamic csms.

        if self._need_recalc_sm:
            t1 = time.process_time()

            n_layers = len(self.layers)
            layersl = list(self.layers.values())

            # todo: thickness=0 layer no need to include, rsp slow

            # exclude the first (incident) layer
            layersl[0].solve()
            ll = layersl[1:]
            nl = n_layers - 1
            _csms = [[(i - 1, j - 1, csm) for (i, j, csm) in li] for li in self.csms[1:]]
            _lm = [a - 1 for a in self._layers_mod if a > 0]

            # calculate csm of uml blocks
            lm = [-1] + _lm
            for i, ilm in enumerate(lm[1:]):
                mp = lm[i] + 1
                if mp < ilm:
                    csm = _csms[mp][-1][2]
                    j = _csms[mp][-1][1] + 1
                    while j < ilm:
                        csm = rsp(*csm, *(_csms[j][-1][2]))
                        _csms[mp].append((mp, j, csm))
                        j = _csms[j][-1][1] + 1
                ll[ilm].solve()
                _csms[ilm].append((ilm, ilm, ll[ilm].sm))

            # handle the first layer
            # Update with fic gap layers: even it is vac its sm is not trivial from vac to fic
            # if layersl[0].is_vac:
            #     self.csms[1:] = [[(i+1, j+1, csm) for (i, j, csm) in li] for li in _csms]
            #     self.csms[0].append((0, 0, self.pr.sm0))
            #     self.csms[0] += [(i-1, j, csm) for (i, j, csm) in self.csms[1]]
            #     layersl[0].csm = layersl[0].sm
            #     layersl[1].csm = self.csms[0][1][2]
            # else:
            self.csms[1:] = [[(i + 1, j + 1, csm) for (i, j, csm) in li] for li in _csms]
            self.csms[0].append((0, 0, layersl[0].sm))
            ss = rsp_sa12lu(*(layersl[0].sm), *(_csms[0][-1][2]))
            self.csms[0].append((0, _csms[0][-1][1]+1, ss))
            layersl[0].csm = layersl[0].sm
            layersl[_csms[0][-1][1]+1].csm = ss

            # handle last layer(s)
            # Update with fic gap layers: even it is vac its sm is not trivial from vac to fic, hence need to remove this `if layersl[-1].is_vac` part
            # if layersl[-1].is_vac:
            #     [li.append((li[-1][0], n_layers - 1, li[-1][2])) for li in self.csms if li[-1][1] == n_layers - 2]
            # elif self._layers_mod[-1] == n_layers-1:
            if self._layers_mod[-1] == n_layers-1:
                s = next(ll[-1] for ll in self.csms if ll[-1][1] == n_layers-2)
                csm = rsp_sb21lu(*(s[2]), *(layersl[-1].sm))
                self.csms[s[0]].append((s[0], n_layers-1, csm))
            else:
                self._calc_csmr_layer(self._layers_mod[-1]+1)

            # from new blocks calc overall sm
            csm = self.csms[0][-1][2]
            j = self.csms[0][-1][1] + 1
            layersl[self.csms[0][-1][1]].csm = csm
            while j < n_layers:
                csm = rsp(*csm, *self.csms[j][-1][2])
                list(self.layers.values())[j].csm = csm
                self.csms[0].append((0, self.csms[j][-1][1], csm))
                j = self.csms[j][-1][1] + 1

            self.sm = self.csms[0][-1][2]
            self._need_recalc_sm = False

            if self.pr.show_calc_time:
                print('{:.6f}   calc sm'.format(time.process_time() - t1))

    def _calc_csm_layer(self, i: int):
        """
        Calculate cumulative scattering matrix from the incident to given layer.

        Note that this subroutine does not do any actual layer solving.

        Parameters
        ----------
        i   :   index of the layer to calculate csm
        """
        t1 = time.process_time()

        layersl = list(self.layers.values())
        if i == len(layersl) - 1:
            warn('csm of the last layer is by definition the overall csm and should have been calculated already.', UserWarning)
        else:
            ii, s = next((j, x) for j, x in enumerate(reversed(self.csms[0])) if x[1] <= i)
            ii = len(self.csms[0]) - ii - 1
            ix = s[1]
            if ix < i:
                ix += 1
                csm = s[2]
                if s[1] == 0:
                    s1 = next(s for s in reversed(self.csms[ix]) if s[1] <= i)
                    csm = rsp_sa12lu(*csm, *s1[2])
                    layersl[s1[1]].csm = csm
                    ii += 1
                    self.csms[0].insert(ii, (0, s1[1], csm))
                    ix = s1[1] + 1
                while ix <= i:
                    s1 = next(s for s in reversed(self.csms[ix]) if s[1] <= i)
                    csm = rsp(*csm, *s1[2])
                    layersl[s1[1]].csm = csm
                    ii += 1
                    self.csms[0].insert(ii, (0, s1[1], csm))
                    ix = s1[1] + 1

        if self.pr.show_calc_time:
            print('{:.6f}   _calc_csm_layer'.format(time.process_time() - t1))

    def _calc_csmr_layer(self, i: int):
        """
        Calculate the cumulative scattering matrix reversed from given layer to the last.

        Note that this subroutine do not do any layer solving.

        Parameters
        ----------
        i   :   index of the layer (forward counting) to calculate csmr till (included).
        """
        t1 = time.process_time()

        n_layers = len(self.layers)
        layersl = list(self.layers.values())

        if i < n_layers:
            if not self.csmsr:
                self.csmsr.append(self.csms[-1][0])
                layersl[-1].csmr = self.csms[-1][0][2]
                # Update with fic gap layers: even it is vac its sm is not trivial from vac to fic, hence need to remove this block
                # if layersl[-1].is_vac:
                #     _csm = (n_layers-2, n_layers-1, self.csms[-2][0][2])
                #     self.csmsr.append(_csm)
                #     layersl[-2].csmr = self.csms[-2][0][2]
                #     if self.csms[-2][-1][1] == n_layers - 2:
                #         self.csms[-2].append(_csm)

            ii, s = next((j, x) for j, x in enumerate(reversed(self.csmsr)) if x[0] >= i)
            ii = len(self.csmsr) - ii - 1
            ix = s[0]

            if ix > i:
                j = i
                _csms = []
                while j < ix:
                    s1 = next(x for x in reversed(self.csms[j]) if x[1] < ix)
                    _csms.append(s1)
                    j = s1[1] + 1

                csm = s[2]
                if ix == n_layers - 1:
                    ss = _csms.pop(-1)
                    csm = rsp_sb21lu(*ss[2], *csm)
                    layersl[ss[0]].csmr = csm
                    self.csms[ss[0]].append((ss[0], n_layers - 1, csm))
                    ii += 1
                    self.csmsr.insert(ii, (ss[0], n_layers - 1, csm))

                for s in reversed(_csms):
                    csm = rsp(*s[2], *csm)
                    layersl[s[0]].csmr = csm
                    self.csms[s[0]].append((s[0], n_layers - 1, csm))
                    ii += 1
                    self.csmsr.insert(ii, (s[0], n_layers - 1, csm))

        if self.pr.show_calc_time:
            print('{:.6f}   _calc_csmr_layer'.format(time.process_time() - t1))

    def _determine_layers(self):
        """Determine if a layer is the output layer or mid layer."""
        for idx, layer in enumerate(self.layers.values()):
            # # inci layer is identified at [adding] time
            # if idx == 0:
            #     if layer.thickness != 0:
            #         warn('You set the first layer (incident region) thickness to be nonzero. This thickness is ignored and set to 0, i.e. treated as infinity. If you meant there was an infinite vacuum before this layer, please explicitly add that using AddLayer().')
            #         layer.thickness = 0.
            #     layer.in_mid_out = 'in'
            if idx == len(self.layers.values()) - 1:
                if layer.thickness != 0:
                    warn('You set the last layer (output region) thickness to be nonzero. This thickness is ignored and set to 0, i.e. treated as infinity. If you meant there was an infinite vacuum after this layer, please explicitly add that using AddLayer().')
                    layer.thickness = 0.
                layer.in_mid_out = 'out'
            elif idx != 0:
                layer.in_mid_out = 'mid'

        # li: Layer = list(self.layers.values())[0]
        # li._set_pr_inci_out()
        # lo: Layer = list(self.layers.values())[-1]
        # lo._set_pr_inci_out()

        # Note: _in_mid_out shouldn't change. If did, layer's sm, al, bl, idx_sm_c_mod, idx_sm_ci_mod all may change.

    def _determine_recalc(self):
        """decide the recalculation token of the subroutines in solver"""
        # t1 = time.process_time()

        n_layers = len(self.layers)
        layersl = list(self.layers.values())

        # collect the indices of the layers that needs recalculation
        self._layers_mod = []
        for i, layer in enumerate(self.layers.values()):
            if layer.if_mod or layer._if_t_change:
                self._need_recalc_sm = True
                self._layers_mod.append(i)

        if self._layers_mod:
            for layer in layersl[self._layers_mod[0]:]:
                layer.csm = None
            for layer in layersl[:self._layers_mod[-1]]:
                layer.csmr = None

        # clear all csm that contains a ml
        if self._need_recalc_sm:
            lm = [-1] + self._layers_mod
            for i, ilm in enumerate(self._layers_mod):
                for j in range(lm[i]+1, ilm+1):
                    if self.csms[j]:
                        iii = next((ii for ii, s in enumerate(self.csms[j]) if s[1] >= ilm), n_layers)
                        del self.csms[j][iii:]
            if self.csmsr:
                iii = next((ii for ii, s in enumerate(self.csmsr) if s[0] <= self._layers_mod[-1]), n_layers)
                del self.csmsr[iii:]

        # update the recalc tokens of ai bo, al bl
        if self._need_recalc_sm:
            self._need_recalc_bi_ao = True
            for layer in self.layers.values():
                layer.need_recalc_al_bl = True

        # print('determine recalc', time.process_time() - t1)

    def solve(self):
        """
        Solve the structure and get s-matrices.

        All user API that require solving will call this method to solve the structure first.
        """
        t1 = time.process_time()

        # if not self.pr.q0_contain_0:
        self._determine_layers()
        self._determine_recalc()
        self._calc_sm()
        # todo: if simulating det(S), no need to calc bi ao al bl
        self._calc_ai_bo_3d()  # need to be after `_calc_sm()` because this needs the eigen of the inci and output layers. todo: determine if need recalc
        self._calc_bi_ao()

        if self.pr.show_calc_time:
            print("{:.6f}   Solving time".format((time.process_time() - t1)))

    def _calc_field_fs_layer_fb(self,
                                layer: str,
                                z: Union[float, List[float], any, Tuple[float]] = None) -> Tuple[any, any, any, any, any, any, any, any, any, any, any, any]:
        """
        Calculate e and h, the Fourier components in a given layer at given z points. return the forward and backward components separately.

        Parameters
        ----------
        layer
        z : array_like
            z is reference to the interface between layer l-1 and l, i.e. 'start' of this layer or the 'left' interface of the layer.
            These points don't need to any particular ordering, neither do they need to be on a regular grid.

        Returns
        -------
        exf, exb, eyf, eyb, ezf, ezb, hxf, hxb, hyf, hyb, hzf, hzb  :   any
            shape (num_g, N) where N is the length of zs. forward and backward field coefficients.

        """
        if z is None:
            za = self.gb.parseData([0.])
        elif hasattr(z, "__len__"):
            za = self.gb.parseData(z)
        else:
            za = self.gb.parseData([z])

        t = self.layers[layer].thickness
        if self.layers[layer].in_mid_out == 'in':
            if (za > 0).any():
                warn('Requesting fields of the incident layer at a position outside the layer. Fields may be diverging.', UserWarning)
        elif self.layers[layer].in_mid_out == 'out':
            if (za < 0).any():
                warn('Requesting fields of the output layer at a position outside the layer. Fields may be diverging.', UserWarning)
        elif (za < 0).any() or (za > t).any():
            warn('Requesting fields of the output layer at a position outside the layer. Fields may be diverging.', UserWarning)

        al, bl = self.layers[layer].al_bl  # for in/out layer?
        phil = self.layers[layer].phil
        psil = self.layers[layer].psil
        qla = self.layers[layer].ql[:, None]  # 1-column 2d array of length 2num_g
        d = self.layers[layer].thickness

        ef = phil * al @ self.gb.exp(1j * qla * za)  # todo: for incident/output region, z too negative and high order ql cause overflow. this is the wave exponential decaying towards the slab
        eb = phil * bl @ self.gb.exp(1j * qla * (d - za))
        hf = psil * al @ self.gb.exp(1j * qla * za)
        hb = -psil * bl @ self.gb.exp(1j * qla * (d - za))
        exf, exb, hxf, hxb = [a[:self.pr.num_g, :] for a in [ef, eb, hf, hb]]
        eyf, eyb, hyf, hyb = [a[self.pr.num_g:, :] for a in [ef, eb, hf, hb]]
        Kx = self.pr.Kx
        Ky = self.pr.Ky
        ezf = 1j / self.omega * self.layers[layer].eizzcm @ (Kx[:, None] * hyf - Ky[:, None] * hxf)
        ezb = 1j / self.omega * self.layers[layer].eizzcm @ (Kx[:, None] * hyb - Ky[:, None] * hxb)
        hzf = 1j / self.omega * self.layers[layer].mizzcm @ (Kx[:, None] * eyf - Ky[:, None] * exf)
        hzb = 1j / self.omega * self.layers[layer].mizzcm @ (Kx[:, None] * eyb - Ky[:, None] * exb)

        return exf, exb, eyf, eyb, ezf, ezb, hxf, hxb, hyf, hyb, hzf, hzb

    def GetAmplitudesByOrder(self,
                      layer: str,
                      z: Union[float, List[float], any, Tuple[float]] = None,
                      order: Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]] = None):
        """
        Get the electric and magnetic fields by Fourier order in a given layer at a given z.

        Parameters
        ----------
        layer
        z
        order:
            order can be int or list of int (for 2d problems), 2-tuple or list of 2-tuples for 3d problems.

        Returns
        -------
        Ex forward, Ex backward, Ey forward, Ey backward, Ez forward, Ez backward, Hx forward, Hx backward, Hy forward, Hy backward, Hz forward, Hz backward
            These are the complex field amplitudes in the given orders.
        """
        if z is None:
            z = 0.

        exf, exb, eyf, eyb, ezf, ezb, hxf, hxb, hyf, hyb, hzf, hzb = self._calc_field_fs_layer_fb(layer=layer, z=z)

        if order is None:
            order = [(0, 0)]
        elif type(order) is int:
            order = [(order, 0)]
        elif type(order) is tuple:
            order = [order]
        elif type(order[0]) is int:
            order = [(o, 0) for o in order]
        idx = self.gb.parseData([self.pr.idx_g.index(o) for o in order])

        result = [f[idx] for f in [exf, exb, eyf, eyb, ezf, ezb, hxf, hxb, hyf, hyb, hzf, hzb]]

        return tuple(result)

    def GetLayerFieldsListPoints(self,
                                 layer: str,
                                 xy: Union[Tuple[float, float], List[Tuple[float, float]]],
                                 z: Union[float, List[float], any, Tuple[float]] = None) -> Tuple[any, any, any, any, any, any]:
        """
        Calculate fields at the list of (x, y) points defined in `xy` in each z plane in the `z` list.

        This is the most versatile possible API for calculating the physical fields at different locations that is still efficient by using numpy array operations.

        Parameters
        ----------
        layer :
            name of the layer
        xy :  a single tuple or a list of tuples.
            This is a list of random (x, y) points. These points don't need to any particular ordering, neither do they need to be on a regular grid.
        z : array_like.
            A list of random z points. These points don't need to any particular ordering, neither do they need to be on a regular grid.

        Returns
        -------
        Ex, Ey, Ez, Hx, Hy, Hz :
            The returned field. Each is a 2d `numpy.ndarray`. The 1st index corresponds to the points in the `xy` list and the 2nd index corresponds to the `z` list.
        """
        if type(xy) is tuple:
            xy = [xy]
        if not hasattr(z, '__len__'):
            z = [z]

        # if not self.pr.q0_contain_0:
        # solve structure first
        self.solve()
        i = list(self.layers.keys()).index(layer)
        self._calc_al_bl_layer(i)

        exf, exb, eyf, eyb, ezf, ezb, hxf, hxb, hyf, hyb, hzf, hzb = self._calc_field_fs_layer_fb(layer, z)  # each has shape (num_g, len(z))
        ex, ey, ez, hx, hy, hz = [a + b for a, b in [(exf, exb), (eyf, eyb), (ezf, ezb), (hxf, hxb), (hyf, hyb), (hzf, hzb)]]

        xa, ya = self.gb.hsplit(self.gb.parseData(xy), 2)  # 2d array with one column

        kxa, kya = self.gb.hsplit(self.gb.parseData(self.pr.ks), 2)  # 2d array with one column

        phase = xa * kxa.T + ya * kya.T  # shape (len(xy), numg)

        Ex, Ey, Ez = [(self.gb.exp(1j * phase) @ e) for e in [ex, ey, ez]]  # shape (len(xy), len(z))

        Hx, Hy, Hz = [(-1j * self.gb.exp(1j * phase) @ h) for h in [hx, hy, hz]]  # shape (len(xy), len(z))

        # else:
        #     Ex, Ey, Ez, Hx, Hy, Hz = [self.gb.nan*self.gb.zeros((len(xy), len(z))) for i in range(6)]

        return Ex, Ey, Ez, Hx, Hy, Hz

    def GetLayerFields(self,
                       layer: str,
                       xmin: float = None,
                       xmax: float = None,
                       nx: int = None,
                       ymin: float = None,
                       ymax: float = None,
                       ny: int = None,
                       zmin: float = None,
                       zmax: float = None,
                       nz: int = None,
                       x: Union[float, List[float], any, Tuple[float]] = None,
                       y: Union[float, List[float], any, Tuple[float]] = None,
                       z: Union[float, List[float], any, Tuple[float]] = None,
                       ) -> Tuple[any, any, any, any, any, any]:
        """
        Get fields at spatial locations (x, y, z) in a layer.

        If x, y and z are lists, then the fields at the 3d grid points defined by these lists are returned.

        Parameters
        ----------
        layer :
            name of the layer.
        xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz :
            the limits (included) and the number of points in each direction. Field values on the 3d grid points spanned will be returned.
            Note `z` is with reference to the interface between previous layer and this one, i.e. the incident-side interface of this layer.
        x, y, z : array_like
            `x` overrides `xmin`, `xmax`, and `nx`. `y` overrides `ymin`, `ymax`, and `ny`. `z` overrides `zmin`, `zmax`, and `nz`.
            Note `z` is with reference to the interface between previous layer and this one, i.e. the incident-side interface of this layer.

        Returns
        -------
        Ex, Ey, Ez, Hx, Hy, Hz :
            Each is an `numpy.ndarray` in Cartesian indexing, i.e. (y, x, z).
        """
        uu = []
        for c, min, max, n, s in zip([x, y, z],
                                     [xmin, ymin, zmin],
                                     [xmax, ymax, zmax],
                                     [nx, ny, nz],
                                     ['x', 'y', 'z']):
            if c is not None:
                if hasattr(c, '__len__'):
                    c = self.gb.parseData(c)
                else:
                    c = self.gb.parseData([c])
                u = c
            elif min is None or max is None or n is None:
                warn(s + " points to get fields not defined properly. Default to 0.")
                u = 0.
            else:
                u = self.gb.linspace(min, max, n)
            uu.append(u)
        x, y, z = uu

        xx, yy = self.gb.meshgrid(x, y)
        xy = list(zip(xx.ravel(), yy.ravel()))

        fields = self.GetLayerFieldsListPoints(layer, xy, z)

        Ex, Ey, Ez, Hx, Hy, Hz = [a.reshape(*(xx.shape), len(z)) for a in fields]

        return Ex, Ey, Ez, Hx, Hy, Hz

    def GetFieldsListPoints(self,
                            xy: Union[Tuple[float, float], List[Tuple[float, float]]],
                            z: Union[float, List[float], any, Tuple[float]] = None) -> Tuple[any, any, any, any, any, any]:
        """
        Calculate fields at the list of (x, y) points defined by `xy` in each z plane in the `z` list.

        This is the most versatile possible API for calculating the physical fields at different locations that is still efficient by using numpy array operations.

        Parameters
        ----------
        xy :  a single tuple or a list of tuples.
            This is a list of random (x, y) points. These points don't need to any particular ordering, neither do they need to be on a regular grid.
        z : array_like.
            A list of random z points. These points don't need to any particular ordering, neither do they need to be on a regular grid.

        Returns
        -------
        Ex, Ey, Ez, Hx, Hy, Hz :
            The returned field. Each is a 2d `numpy.ndarray`. The 1st index corresponds to the points in the `xy` list and the 2nd index corresponds to the `z` list.

        """
        self.solve()

        ll = list(self.layers.keys())

        if hasattr(z, "__len__"):
            za = self.gb.parseData(z)
        else:
            za = self.gb.parseData([z])

        Fields = [self.gb.zeros((len(xy), len(za)), dtype=self.gb.complex128) for i in range(6)]

        z_interfaces = self.thicknesses_c[:-1]  # -1 is output with thickness 0

        i_in_l = [za < z_interfaces[0]]
        for idx, zi in enumerate(z_interfaces[:-1]):
            i_in_l.append((za >= zi) * (za < z_interfaces[idx + 1]))
        i_in_l.append(za >= z_interfaces[-1])

        for idx, iin in enumerate(i_in_l):
            za_l = za[iin] - ([0] + z_interfaces)[idx]  # z coordinate of this layer w.r.t. the left interface of this layer
            z_l = za_l.tolist()
            if z_l:
                fields = self.GetLayerFieldsListPoints(ll[idx], xy, z_l)
                for F, f in zip(Fields, fields):
                    F[:, iin] = f

        Ex, Ey, Ez, Hx, Hy, Hz = Fields

        return Ex, Ey, Ez, Hx, Hy, Hz

    def GetFields(self,
                  xmin: float = None,
                  xmax: float = None,
                  nx: int = None,
                  ymin: float = None,
                  ymax: float = None,
                  ny: int = None,
                  zmin: float = None,
                  zmax: float = None,
                  nz: int = None,
                  x: Union[float, List[float], any, Tuple[float]] = None,
                  y: Union[float, List[float], any, Tuple[float]] = None,
                  z: Union[float, List[float], any, Tuple[float]] = None,
                  ) -> Tuple[any, any, any, any, any, any]:
        """
        Calculate fields on a 3D grid.

        Parameters
        ----------
        xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz :
            the limits (included) and the number of points in each direction. Field values on the 3d grid points spanned will be returned.
        x, y, z : array_like
            `x` overrides `xmin`, `xmax`, and `nx`. `y` overrides `ymin`, `ymax`, and `ny`. `z` overrides `zmin`, `zmax`, and `nz`.

        Returns
        -------
        Ex, Ey, Ez, Hx, Hy, Hz :
            Each is an `numpy.ndarray` in Cartesian indexing, i.e. (y, x, z).
        """

        uu = []
        for c, min, max, n, s in zip([x, y, z],
                                     [xmin, ymin, zmin],
                                     [xmax, ymax, zmax],
                                     [nx, ny, nz],
                                     ['x', 'y', 'z']):
            if c is not None:
                if hasattr(c, '__len__'):
                    c = self.gb.parseData(c)
                else:
                    c = self.gb.parseData([c])
                u = c
            elif min is None or max is None or n is None:
                warn(s + " points to get fields not defined properly. Default to 0.")
                u = 0.
            else:
                u = self.gb.linspace(min, max, n)
            uu.append(u)

        x, y, z = uu
        xx, yy = self.gb.meshgrid(x, y)
        xy = list(zip(xx.ravel(), yy.ravel()))

        fields = self.GetFieldsListPoints(xy, z)

        Ex, Ey, Ez, Hx, Hy, Hz = [a.reshape(*(xx.shape), len(z)) for a in fields]

        return Ex, Ey, Ez, Hx, Hy, Hz

    def GetPowerFlux(self,
                     layer: str,
                     z: Union[float, List[float]] = None
                     ) -> Tuple[Union[float, any], Union[float, any]]:
        """
        get the power flux in z direction in a layer at the given `z` points.

        Parameters
        ----------
        layer :
            name of the layer
        z : array_like
            a list of random points. These points don't need to any particular ordering, neither do they need to be on a regular grid.

        Returns
        -------
        sf, sb :
            forward and backward power flux. Each is an 1d `numpy.ndarray` of length equal to the number of `z` points.

        """

        # if not self.pr.q0_contain_0:

        self.solve()
        i = list(self.layers.keys()).index(layer)
        self._calc_al_bl_layer(i)

        t1 = time.process_time()
        exf, exb, eyf, eyb, ezf, ezb, hxf, hxb, hyf, hyb, hzf, hzb = self._calc_field_fs_layer_fb(layer, z)
        ex, ey, ez, hx, hy, hz = [a + b for a, b in [(exf, exb), (eyf, eyb), (ezf, ezb), (hxf, hxb), (hyf, hyb), (hzf, hzb)]]

        sf = -1.j / 4. * ((self.gb.einsum('i...,i...', ex.conj(), hyf) - self.gb.einsum('i...,i...', ey.conj(), hxf))
                          - (self.gb.einsum('i...,i...', hy.conj(), exf) - self.gb.einsum('i...,i...', hx.conj(), eyf)))  # 1d array of length len(z)
        sb = -1.j / 4. * ((self.gb.einsum('i...,i...', ex.conj(), hyb) - self.gb.einsum('i...,i...', ey.conj(), hxb))
                          - (self.gb.einsum('i...,i...', hy.conj(), exb) - self.gb.einsum('i...,i...', hx.conj(), eyb)))

        if sf.size == 1:
            sf = sf[0].real
            sb = sb[0].real
        else:
            sf = sf.real
            sb = sb.real

        if self.pr.show_calc_time:
            print('{:.6f}   calc flux from coef'.format(time.process_time() - t1))

        # else:
        #     sf = float('nan')
        #     sb = float('nan')

        return sf, sb

    def GetPowerFluxByOrder(self,
                            layer: str,
                            order: Union[int, List[int], Tuple[int, int], List[Tuple[int, int]]],
                            z: Union[float, List[float]] = None) -> Tuple[any, any]:
        """
        Get the power flux of given orders defined by `order` in z direction in a layer at the given `z` points.

        Parameters
        ----------
        layer :
            name of the layer
        order :
            Fourier order defined by (i, j) or a list of (i, j) where i and j are integers.
        z : array_like
            a list of random points. These points don't need to any particular ordering, neither do they need to be on a regular grid.

        Returns
        -------
        sf, sb :    ndarray
            forward and backward power flux. Each is an 2d ndarray of shape (len(order), len(z)）.

        See Also
        --------
        GetPowerFlux

        """

        # if not self.pr.q0_contain_0:

        self.solve()
        i = list(self.layers.keys()).index(layer)
        self._calc_al_bl_layer(i)

        t1 = time.process_time()
        if order is None:
            order = [(0, 0)]
        elif type(order) is int:
            order = [(order, 0)]
        elif type(order) is tuple:
            order = [order]
        elif type(order[0]) is int:
            order = [(o, 0) for o in order]
        else:
            raise('Incorrect datatype for input argument `order`.', RuntimeError)
        idx = self.gb.parseData([self.pr.idx_g.index(o) for o in order])

        exf, exb, eyf, eyb, ezf, ezb, hxf, hxb, hyf, hyb, hzf, hzb = self._calc_field_fs_layer_fb(layer, z)

        ex, ey, hx, hy = [a + b for a, b in [(exf, exb), (eyf, eyb), (hxf, hxb), (hyf, hyb)]]

        sf = -1.j / 4. * ((ex.conj()[idx, :] * hyf[idx, :] - ey.conj()[idx, :] * hxf[idx, :])
                          - (hy.conj()[idx, :] * exf[idx, :] - hx.conj()[idx, :] * eyf[idx, :]))  # 1d array of length len(z)
        sb = -1.j / 4. * ((ex.conj()[idx, :] * hyb[idx, :] - ey.conj()[idx, :] * hxb[idx, :])
                          - (hy.conj()[idx, :] * exb[idx, :] - hx.conj()[idx, :] * eyb[idx, :]))

        if sf.size == 1:
            sf = sf.ravel()[0].real
            sb = sb.ravel()[0].real
        else:
            sf = sf.real
            sb = sb.real

        if self.pr.show_calc_time:
            print('{:.6f}   calc flux from coef'.format(time.process_time() - t1))

        # else:
        #     sf = float('nan')
        #     sb = float('nan')

        return sf, sb

    def GetSMatrix(self,
                   radiation_channels_only: bool = True,
                   channels: List[Tuple[int, int]] = None,
                   channels_in: List[Tuple[int, int]] = None,
                   channels_out: List[Tuple[int, int]] = None,
                   channels_exclude: List[Tuple[int, int]] = None
                   ) :
        """
        Get the scattering matrix of the entire structure.

        Can optionally select a few channels and return the scattering matrix between them.

        Parameters
        ----------
        radiation_channels_only :
            if or not to only include radiation channels in the incident and output regions
        channels :
            selected channels in the input and the transmission regions to calculate det.
            overrides radiation_channels_only.
        channels_in :
            the selected channels in the incident region
        channels_out :
            the selected channels in the output region
            if either channels_in or channels_out is specified, `channels` will be overridden.
        channels_exclude:
            select all other channels but these ones in both the incident and the transmission regions.
            Overrides `channels`, `channels_in` and `channels_out`.

        Returns
        -------
        dets :
            the scattering matrix.
        idx :
            the channel index
        """

        # if not self.pr.q0_contain_0:

        self.solve()
        sm_b = self.sm

        sm = self.gb.block([[sm_b[0], sm_b[1]],
                       [sm_b[2], sm_b[3]]])

        rci = []
        rco = []
        _rco = []

        if channels_exclude is not None:
            rci = list(range(self.pr.num_g))
            for a in channels_exclude:
                i = self.pr.idx_g.index(a)
                rci.pop(i)
            rci += [a + self.pr.num_g for a in rci]
            rco = [a + 2 * self.pr.num_g for a in rci]
        elif channels_in is not None:
            rci = [self.pr.idx_g.index(a) for a in channels_in]
            rci += [a+self.pr.num_g for a in rci]
            if channels_out is not None:
                rco = [self.pr.idx_g.index(a) + 2 * self.pr.num_g for a in channels_out]
                rco += [a + self.pr.num_g for a in rco]
        elif channels_out is not None:
            rco = [self.pr.idx_g.index(a) + 2 * self.pr.num_g for a in channels_out]
            rco += [a + self.pr.num_g for a in rco]
        elif channels is not None:
            rci = [self.pr.idx_g.index(a) for a in channels]
            rci += [a+self.pr.num_g for a in rci]
            rco = [a + 2 * self.pr.num_g for a in rci]
        elif radiation_channels_only:
            layersl = list(self.layers.values())
            rci = layersl[0].rad_cha
            rco = layersl[-1].rad_cha
            _rco = [a + 2*self.pr.num_g for a in rco]

        if rci or _rco:
            rc = rci + _rco
            sm = sm[rc, :][:, rc]

        if rci:
            idx_i = [self.pr.idx_g[ii] for ii in rci[:len(rci)//2]]
        else:
            idx_i = self.pr.idx_g
        if rco:
            idx_o = [self.pr.idx_g[ii] for ii in rco[:len(rco)//2]]
        else:
            idx_o = self.pr.idx_g

        # else:
        #     sm = float('nan')
        #     idx_i = float('nan')
        #     idx_o = float('nan')

        return sm, (idx_i, idx_o)

    def GetSMatrixDet(self,
                      radiation_channels_only: bool=False,
                      channels: List[Tuple[int, int]] = None,
                      channels_in: List[Tuple[int, int]] = None,
                      channels_out: List[Tuple[int, int]] = None,
                      channels_exclude: List[Tuple[int, int]] = None
                      ) -> Tuple[Union[float, complex], float]:
        """
        Calculate the determinant of the scattering matrix.

        Because the determinant typically leads to overflow or underflow, the sign and the natural logarithm of the determinant are returned.

        Can optionally select a few channels and calculate the determinant of the scattering matrix among just these channels.

        Parameters
        ----------
        radiation_channels_only :
            if or not to only include radiation channels in the incident and output regions
        channels :
            selected channels in the input and the transmission regions to calculate det.
            overrides radiation_channels_only.
        channels_in :
            the selected channels in the incident region
        channels_out :
            the selected channels in the output region
            if either channels_in or channels_out is specified, `channels` will be overridden.
        channels_exclude:
            select all other channels but these ones in both the incident and the transmission regions.
            Overrides `channels`, `channels_in` and `channels_out`.

        Returns
        -------
        dets :
            the sign and the natural log of the determinant of the scattering matrix.
        """
        sm, idx = self.GetSMatrix(radiation_channels_only=radiation_channels_only, channels=channels, channels_in=channels_in, channels_out=channels_out, channels_exclude=channels_exclude)

        if sm.any():
            dets = la.slogdet(sm)
        else:
            dets = float('nan')

        return dets

    def GetSMatrixDeterminant(self, *args, **kwargs):
        warn("This method is an alias of `GetSMatrixDet()`. It's recommended to use `GetSMatrixDet()` instead.", FutureWarning)
        return self.GetSMatrixDet(*args, **kwargs)

    def GetReducedSMatrixDeterminant(self) -> Tuple[Union[float, complex], float]:
        """
        Calculate the determinant of the reduced scattering matrix that contains only the radiation channels.

        The reduced scattering matrix is defined as taking only the elements that correspond to radiative channels from the entire scattering matrix.

        See Also
        --------
        GetSMatrixDet

        Returns
        -------
        dets :
            the sign and the natural log of the determinant of the scattering matrix.
        """
        warn("This method is deprecated. Use `GetRadiativeSMatrixDet(radiation_channels_only=True)` instead.", FutureWarning)
        return self.GetSMatrixDet(radiation_channels_only=True)

    def GetRadiativeSMatrixDet(self) -> Tuple[Union[float, complex], float]:
        """
        Calculate the determinant of the reduced scattering matrix that contains only the radiation channels.

        The reduced scattering matrix is defined as taking only the elements that correspond to radiative channels from the entire scattering matrix.

        See Also
        --------
        GetSMatrixDet

        Returns
        -------
        dets :
            the sign and the natural log of the determinant of the scattering matrix.
        """
        return self.GetSMatrixDet(radiation_channels_only=True)
    
