# -*- coding: utf-8 -*-
from GenericBackend import genericBackend as gb
# import numpy.linalg as la
from inkstone.shps import Rect, Para, Disk, Elli, Poly, OneD
from warnings import warn
from typing import Tuple, Union, List, Optional
from inkstone.mtr import Mtr


class Bx:
    # Bx doesn't know what lattice it is in

    def __init__(self, mtr, shp, name=None, outside=None,gb=gb, **kwargs):
        """
        A box.
        The box has a name, its `mtr`, its `shp`.

        Can `set_shape`, set `ks`, and get Fourier series of epsi and mu by `fs()`.

        Parameters
        ----------
        mtr         :   Mtr
                        material
        shp         :   {'rectangle', 'disk', 'ellipse', 'polygon', 'parallelogram', '1d'}
        name        :   str
        outside     :   Bx
                        which bx contains this one.
        kwargs      :
                        other arguments for setting up the shape, to be passed on to `Rect`, `Disk`, `Elli`, `Poly`, or `OneD`.
        """

        self.epsi_ft = None
        self.epsi_inv_ft = None
        self.mu_ft = None
        self.mu_inv_ft = None

        self.mtr = mtr
        self.name = name

        if shp == 'rectangle':
            self.shp = Rect(**kwargs)
        elif shp == 'parallelogram':
            self.shp = Para(**kwargs)
        elif shp == 'disk':
            self.shp = Disk(**kwargs)
        elif shp == 'polygon':
            self.shp = Poly(**kwargs)
        elif shp == 'ellipse':
            self.shp = Elli(**kwargs)
        elif shp == '1d':
            self.shp = OneD(**kwargs)
        else:
            warn('shape not recognized.', UserWarning)

        self.outside: Bx = outside

    @property
    def shape(self):
        return self.shp.shape

    @property
    def ks(self):
        """Return the list of k points on which Fourier transform coefficients are calculated"""
        return self.shp.ks

    @ks.setter
    def ks(self, val):
        # After you set the ks here, Bx doesn't know if outside ks has changed. If it has, then you need to do ft(ks) to give the new ks.
        self.shp.ks = val

    def ft(self, ks=None, **kw_gibbs) -> Tuple[any, any, any, any]:
        """
        Return the Fourier transform of the shape. The FT of a shape is understood as a function that is 1 inside and 0 outside.
        You can choose whether to have Gibbs correction for the returned FT coefficients.

        Parameters
        ----------
        ks          :   list
        kw_gibbs    :   keyword arguments related to Gibbs correction, to be passed on to `_calc_ft`

        Returns
        -------

        """
        if ks:
            self.ks = ks
        ep, ei, mu, mi = self._calc_ft(**kw_gibbs)
        self.epsi_ft = ep
        self.epsi_inv_ft = ei
        self.mu_ft = mu
        self.mu_inv_ft = mi

        return self.epsi_ft, self.epsi_inv_ft, self.mu_ft, self.mu_inv_ft

    def _calc_ft(self, **kw_gibbs) -> Tuple[any, any, any, any]:
        """
        Calculate the Fourier transform of the box.

        Parameters
        ----------
        kw_gibbs    :   keyword arguments related to Gibbs correction, to be passed on to `.shps.Shp.ft`

        Returns
        -------

        """
        # t0 = time.process_time()
        _ft = gb.parseData(self.shp.ft(**kw_gibbs), dtype=gb.complex128)
        # print(time.process_time()-t0)

        # no way of knowing if its material or that of the outer box changed
        # hence just recalculate every time. It takes no time anyway.
        epsi = self.mtr.epsi
        epsi_inv = self.mtr.epsi_inv
        mu = self.mtr.mu
        mu_inv = self.mtr.mu_inv
        if self.outside:
            epsi_out = self.outside.mtr.epsi
            epsi_out_inv = self.outside.mtr.epsi_inv
            mu_out = self.outside.mtr.mu
            mu_out_inv = self.outside.mtr.mu_inv
        else:
            epsi_out = gb.eye(3, dtype=gb.complex128)
            epsi_out_inv = gb.eye(3, dtype=gb.complex128)
            mu_out = gb.eye(3, dtype=gb.complex128)
            mu_out_inv = gb.eye(3, dtype=gb.complex128)
        epsi = epsi - epsi_out  # shouldn't do epsi -= epsi_out, which would be in-place change, which will change epsi of the material.
        epsi_inv = epsi_inv - epsi_out_inv
        mu = mu_inv - mu_out
        mu_inv = mu_inv - mu_out_inv

        # t1 = time.process_time()
        ep = epsi[None, :, :] * _ft[:, None, None]  # (N, 3, 3) shape
        ei = epsi_inv[None, :, :] * _ft[:, None, None]
        mu = mu[None, :, :] * _ft[:, None, None]
        mi = mu_inv[None, :, :] * _ft[:, None, None]
        # t2 = time.process_time()
        # print('ndarray time', t2-t1)
        # t1 = time.process_time()
        ep, ei, mu, mi = [[a for a in em] for em in [ep, ei, mu, mi]]
        # print('convert to list time', time.process_time()-t1)

        return ep, ei, mu, mi

    def set_shape(self, width=None, center=None, angle=None, shear_angle=None, half_lengths=None, side_lengths=None, radius=None, vertices=None, **kw_gibbs):
        """
        Set the shape geometry.

        Parameters
        ----------
        width           :   float
        center          :   tuple[float, float]
        angle           :   float
        shear_angle     :   float
        half_lengths    :   tuple[float, float]
        side_lengths    :   tuple[float, float]
        radius          :   float
        vertices        :   list[tuple[float, float]]
        kw_gibbs        :   other parameters for setting gibbs correction

        """
        # todo: what self.shp is is conditional; when refactoring this is a problem.
        if self.shp.shape == 'rectangle':
            if center is not None:
                self.shp.center = center
            if angle is not None:
                self.shp.angle = angle
            if side_lengths is not None:
                self.shp.side_lengths = side_lengths
        if self.shp.shape == 'parallelogram':
            if center is not None:
                self.shp.center = center
            if angle is not None:
                self.shp.angle = angle
            if shear_angle is not None:
                self.shp.shear_angle = shear_angle
            if side_lengths is not None:
                self.shp.side_lengths = side_lengths
        if self.shp.shape == 'disk':
            if center is not None:
                self.shp.center = center
            if radius is not None:
                self.shp.radius = radius
        if self.shp.shape == 'ellipse':
            if center is not None:
                self.shp.center = center
            if angle is not None:
                self.shp.angle = angle
            if half_lengths is not None:
                self.shp.half_lengths = half_lengths
        if self.shp.shape == 'polygon':
            if vertices is not None:
                self.shp.vertices = vertices
        if self.shp.shape == '1d':
            if width is not None:
                self.shp.width = width
            if center is not None:
                self.shp.center = center

        self.shp.use_gibbs_correction(**kw_gibbs)


