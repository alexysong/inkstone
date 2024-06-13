# -*- coding: utf-8 -*-

from GenericBackend import genericBackend as gb
from warnings import warn
import copy
from typing import List, Union, Tuple, Optional
from inkstone.ft.ft_1d_sq import ft_1d_sq
from inkstone.ft.ft_2d_rct import ft_2d_rct
from inkstone.ft.ft_2d_para import ft_2d_para
from inkstone.ft.ft_2d_ellip import ft_2d_ellip
from inkstone.ft.ft_2d_disk import ft_2d_disk
from inkstone.ft.ft_2d_poly import ft_2d_poly
from inkstone.ft.poly_area import poly_area
from inkstone.ft.gibbs import gibbs_corr


class Shp:
    # Shp doesn't know what lattice it is in
    # Shape recalculate its fourier series at read time (when accessing self.ft()).
    # FT calculation is analytic hence quick, recalculate every time ft() is called. No need to check if parameters modified

    def __init__(self,
                 shp: str,
                 ks: List[Union[float, Tuple[float, float]]] =None,
                 gb=gb,
                 **kw_gibbs):
        """
        Basic shape.

        You can specify the geometry of the shape, and get the Fourier transform (FT) of the shape. The FT of a shape is understood as a function that is 1 inside and 0 outside.

        You can choose whether to have Gibbs correction for the returned FT coefficients.

        You can give the list of k points `ks` at initialization time, use `Shp.ks` to set it, or supply it at `Shp.ft(ks)`. with `ks` given, you can call `Shp.ft()` to get the Fourier transform on the k points.

        Parameters
        ----------
        shp                 :   what kind of shape this is
        ks                  :   list of k points to calculate FT on
        kw_gibbs            :   keyword arguments for Gibbs correction, to be passed to `.ft.gibbs.gibbs_corr`
        """
        self.shape: str = shp
        self.area: Optional[float] = None
        self._ft: Optional[List[float]] = None  # 1d array Fourier series at the k points
        self._if_gibbs_corr = True
        self._kw_gibbs: dict = {}

        self.ks = ks  # list of k points to calculate Fourier series
        self.use_gibbs_correction(**kw_gibbs)
        
        self.gb = gb

    @property
    def ks(self) -> List[Union[float, Tuple[float, float]]]:
        """The list of 1d k points or 2d (kx, ky) tuples on which the Fourier coefficients of the shape is calculated."""
        return self._ks

    @ks.setter
    def ks(self, val: List[Union[float, Tuple[float, float]]]):
        self._ks: List[Union[float, Tuple[float, float]]] = val  # don't do deep copy which would take longer time than the actual Fourier transform.

    def use_gibbs_correction(self, if_gibbs_correction: bool = None, **kw_gibbs):
        """
        Choose whether to have correction for Gibbs phenomenon.
        Parameters
        ----------
        if_gibbs_correction :   whether to use gibbs correction when calculating the Fourier transform
        kw_gibbs            :   other keyword arguments for Gibbs correction, to be passed on to `.ft.gibbs.gibbs_corr`
        -------

        """
        if if_gibbs_correction is not None:
            self._if_gibbs_corr = if_gibbs_correction
        for key, value in kw_gibbs.items():
            self._kw_gibbs[key] = value

    def ft(self,
           ks: Optional[List[Tuple[float, float]]] = None,
           **kw_gibbs
           ) -> Optional[List[complex]]:
        """
        Get the Fourier transform coefficients on the given k points in `ks`.
        Can choose to have correction for Gibbs phenomenon.

        Parameters
        ----------
        ks          :   list of k points to calculate FT on
        kw_gibbs    :   keyword arguments for Gibbs correction, to be passed on to `.ft.gibbs.gibbs_corr`.
        """
        if ks:
            self.ks: List = ks

        self._ft = self._calc_ft()
        s = 1
        self.use_gibbs_correction(**kw_gibbs)
        if self._if_gibbs_corr:
            s = gibbs_corr(self.ks, **self._kw_gibbs)
        ft = (self.gb.parseData(self._ft, dtype=self.gb.complex128) * self.gb.parseData(s, dtype=self.gb.complex128)).tolist()
        self._ft = ft
        return self._ft

    def _calc_ft(self) -> List[complex]:
        """
        Get fourier coefficients at given k points
        To be implemented in different shapes.
        """

    use_gibbs_correction.__doc__ += gibbs_corr.__doc__
    __init__.__doc__ += use_gibbs_correction.__doc__
    __init__.__doc__ += ks.__doc__
    ft.__doc__ += ks.__doc__
    ft.__doc__ += use_gibbs_correction.__doc__


class OneD(Shp):
    def __init__(self, width: float, center: float = None, **kwargs):
        """
        This is a 1D shape.

        If you query the Fourier transform coefficients at a list of 2D k points (kx, ky), the first nonzero k components will be retained while the other be ignored. For example, [(0, 1), (2, 3)] effectively become [(0, 1), (0, 3)]

        The area of this shape is defined as its width.
        """
        super(OneD, self).__init__('1d', **kwargs)

        self._center = None

        self.width = width
        self.center = center

    @property
    def width(self) -> float:
        """The width of the 1d shape."""
        return self._width

    @width.setter
    def width(self, val: float):
        self._width = val
        self.area = val

    @property
    def center(self) -> float:
        """The center of the 1d shape"""
        return self._center

    @center.setter
    def center(self, val: float):
        if val is None:
            if self._center is None:
                self._center = 0.
        else:
            self._center = val

    @property
    def ks(self) -> List[float]:
        """The list of 1d k points or 2d (kx, ky) tuples on which the Fourier coefficients of the shape is calculated."""
        return self._ks

    @ks.setter
    def ks(self, val: List[Union[float, Tuple[float, float]]]):
        if val:
            if type(val[0]) is tuple:
                for v in val:
                    if v[0] != 0:
                        self._ks: List[float] = [k[0] for k in val]
                        break
                    elif v[1] != 0:
                        self._ks: List[float] = [k[1] for k in val]
                        break
                self._ks: List[float] = [k[0] for k in val]
            else:
                self._ks: List[float] = val

    def _calc_ft(self) -> List[complex]:
        return ft_1d_sq(self.width, self.ks, self.center)


class Rect(Shp):
    def __init__(self, side_lengths, center=None, angle=None, **kwargs):
        """
        A rectangular shape.

        Parameters
        ----------
        side_lengths    :   tuple[float, float]
        center          :   tuple[float, float]
        angle           :   float
        """

        super(Rect, self).__init__('rectangle', **kwargs)

        self._center = None
        self._angle = None

        self.side_lengths = side_lengths
        self.center = center
        self.angle = angle

    @property
    def side_lengths(self):
        return self._side_lengths

    @side_lengths.setter
    def side_lengths(self, val):
        # assume when setting side lengths it must have been changed. no `If` which is slow.
        self.area = val[0] * val[1]
        self._side_lengths = val

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        if val is None:
            if self._center is None:
                self._center = (0., 0.)
        else:
            self._center = val

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, val):
        if val is None:
            if self._angle is None:
                self._angle = 0.
        else:
            self._angle = val

    def _calc_ft(self) -> List[complex]:
        """
        Get fourier coefficients at given k points
        """
        return ft_2d_rct(self.side_lengths[0], self.side_lengths[1], self.ks, self.center, self.angle)


class Para(Shp):
    def __init__(self, side_lengths, center=None, shear_angle=None, angle=None, **kwargs):
        """
        A parallelogram shape.

        This shape has two actual side lengths defined here.

        Can define the shear angle, i.e. the included angle between the two arms.

        Then the parallelogram can be rotated by `angle` and then shifted to new `center`.

        Parameters
        ----------
        side_lengths    :   tuple[float, float]
                            actual side lengths of the shape
        center          :   tuple[float, float]
        shear_angle     :   float
                            the angle between the two side arms in degrees.
        angle           :   float
                            the rotation angle of the entire shape in degrees

        """

        super(Para, self).__init__('parallelogram', **kwargs)

        self.side_lengths = side_lengths

        self._center = None
        self._shear_angle = None
        self._angle = None

        self.center = center
        self.angle = angle
        self.shear_angle = shear_angle

    @property
    def side_lengths(self):
        return self._side_lengths

    @side_lengths.setter
    def side_lengths(self, val):
        # assume when setting side lengths it must have been changed. no `If` which is slow.
        self.area = val[0] * val[1]
        self._side_lengths = val

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        if val is None:
            if self._center is None:
                self._center = (0., 0.)
        else:
            self._center = val

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, val):
        if val is None:
            if self._angle is None:
                self._angle = 0.
        else:
            self._angle = val

    @property
    def shear_angle(self):
        return self._shear_angle

    @shear_angle.setter
    def shear_angle(self, val):
        if val is None:
            if self._shear_angle is None:
                self._shear_angle = 0.
        else:
            self._shear_angle = val

    def _calc_ft(self) -> List[complex]:
        """
        Get fourier coefficients at given k points
        """
        return ft_2d_para(a=self.side_lengths[0], b=self.side_lengths[1], ks=self.ks, center=self.center, shear_angle=self.shear_angle, rotate_angle=self.angle)


class Elli(Shp):
    def __init__(self, half_lengths, center=None, angle=None, **kwargs):
        """
        A ellipse shape.

        Parameters
        ----------
        half_lengths    :   tuple[float, float]
        center          :   tuple[float, float]
        angle           :   float
        """

        super(Elli, self).__init__('ellipse', **kwargs)

        self.half_lengths = half_lengths

        self._center = None
        self._angle = None

        self.center = center
        self.angle = angle

    @property
    def half_lengths(self):
        return self._half_lengths

    @half_lengths.setter
    def half_lengths(self, val):
        # assume when setting side lengths it must have been changed. no `If` which is slow.
        self._half_lengths = val
        self.area = self.gb.pi * val[0] * val[1]

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        if val is None:
            if self._center is None:
                self._center = (0., 0.)
        else:
            self._center = val

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, val):
        if val is None:
            if self._angle is None:
                self._angle = 0.
        else:
            self._angle = val

    def _calc_ft(self) -> List[complex]:
        """
        Get fourier coefficients at given k points
        """
        return ft_2d_ellip(self.half_lengths[0], self.half_lengths[1], self.ks, self.center, self.angle)


class Disk(Shp):

    def __init__(self, radius, center=None, **kwargs):
        """
        A disk shape.
        """
        super(Disk, self).__init__('disk', **kwargs)

        self.radius = radius

        self._center = None

        self.center = center

    @property
    def radius(self):
        """Return the radius of the disk."""
        return self._radius

    @radius.setter
    def radius(self, val):
        # assume when setting side lengths it must have been changed. no `If` which is slow.
        self._radius = val
        self.area = self.gb.pi * val ** 2

    @property
    def center(self):
        """Return the center of the disk."""
        return self._center

    @center.setter
    def center(self, val):
        if val is None:
            if self._center is None:
                self._center = (0., 0.)
        else:
            self._center = val

    def _calc_ft(self) -> List[complex]:
        """
        Get fourier coefficients at given k points
        """
        return ft_2d_disk(self.radius, self.ks, self.center)


class Poly(Shp):
    def __init__(self, vertices, **kwargs):
        """
        A polygon shape

        Parameters
        ----------
        vertices        :   list[tuple[float, float]]
        """

        super(Poly, self).__init__('polygon', **kwargs)

        self.vertices = vertices

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, val):
        # assume when setting side lengths it must have been changed. no `If` which is slow.
        self._vertices = val
        self.area = poly_area(val)

    def _calc_ft(self) -> List[complex]:
        """
        Get fourier coefficients at given k points
        """
        return ft_2d_poly(self.vertices, self.ks)

