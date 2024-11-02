import warnings
from warnings import warn

import numpy as np
import scipy.linalg as sla
import scipy.special as sp
import scipy.fft as sfft
from inkstone.backends.Backend import Backend


class NumpyBackend(Backend):

    def __init__(self):

        super().__init__()
        self.raw_type = np.ndarray
        self.complex128 = np.complex128
        self.float64 = np.float64
        self.int32 = np.int32
        self.pi = np.pi

        self.ifftshift = np.fft.ifftshift

    def abs(self, *args, **kwargs):
        return np.abs(*args, **kwargs)

    def arange(self, *args, **kwargs):
        return np.arange(*args, **kwargs)

    def arccos(self, *args, **kwargs):
        return np.arccos(*args, **kwargs)

    def arcsin(self, *args, **kwargs):
        return np.arcsin(*args, **kwargs)

    def ceil(self, *args, **kwargs):
        return np.ceil(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        return np.concatenate(*args, **kwargs)

    def conj(self, *args, **kwargs):
        return np.conj(*args, **kwargs)

    def cos(self, *args, **kwargs):
        return np.cos(*args, **kwargs)

    def diag(self, *args, **kwargs):
        return np.diag(*args, **kwargs)

    def dot(self, *args, **kwargs):
        return np.dot(*args, **kwargs)

    def einsum(self, *args, **kwargs):
        return np.einsum(*args, **kwargs)

    def exp(self, *args, **kwargs):
        return np.exp(*args, **kwargs)

    def eye(self, *args, **kwargs):
        return np.eye(*args, **kwargs)

    def full(self, *args, **kwargs):
        return np.full(*args, **kwargs)

    def hsplit(self, *args, **kwargs):
        return np.hsplit(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return np.linspace(*args, **kwargs)

    def logical_not(self, *args, **kwargs):
        return np.logical_not(*args, **kwargs)

    def lu_factor(self, *args, **kwargs):
        return sla.lu_factor(*args, **kwargs)

    def maximum(self, *args, **kwargs):
        return np.maximum(*args, **kwargs)

    def moveaxis(self, *args, **kwargs):
        return np.moveaxis(*args, **kwargs)

    def ones(self, *args, **kwargs):
        return np.ones(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        return np.repeat(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return np.reshape(*args, **kwargs)

    def roll(self, *args, **kwargs):
        return np.roll(*args, **kwargs)

    def sin(self, *args, **kwargs):
        return np.sin(*args, **kwargs)

    def sinc(self, *args, **kwargs):
        return np.sinc(*args, **kwargs)

    def slogdet(self, *args, **kwargs):
        return np.linalg.slogdet(*args, **kwargs)

    def solve(self, *args, **kwargs):
        return sla.solve(*args, **kwargs)

    def square(self, *args, **kwargs):
        return np.square(*args, **kwargs)

    def sqrt(self, *args, **kwargs):
        return np.sqrt(*args, **kwargs)

    def sum(self, *args, **kwargs):
        return np.sum(*args, **kwargs)

    def tan(self, *args, **kwargs):
        return np.tan(*args, **kwargs)

    def where(self, *args, **kwargs):
        return np.where(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return np.zeros(*args, **kwargs)

    def stack(self, *args, **kwargs):
        return np.stack(*args, **kwargs)

    def j0(self, *args, **kwargs):
        return sp.j0(*args, **kwargs)

    def j1(self, *args, **kwargs):
        return sp.j1(*args, **kwargs)

    def eig(self, A):
        return sla.eig(A)

    def data(self, i: any, dtype=None, **kwargs):
        return np.array(i, dtype=dtype)

    def cross(self, a, b, dim=None):
        return np.cross(a, b)

    def meshgrid(self, *xi):
        return np.meshgrid(*xi)

    def castType(self, i, typ):  # typ(e), avoid collision with keyword
        return i.astype(typ)

    def parseList(self, tup):
        return np.array(tup)

    def laCross(self, a, b):
        return np.cross(a, b)

    def getSize(self, i):
        return i.size

    def clone(self, i):
        return i.copy()

    def triu_indices(self, row, offset=0, col=None):
        if not col:
            col = row
        return np.triu_indices(row, offset, col)

    def lu_solve(self, p, q):
        return sla.lu_solve(p, q)

    def ones(self, p, dtype=np.float64):
        return np.ones(p, dtype=dtype)

    def zeros(self, p, dtype=np.float64):
        return np.zeros(p, dtype=dtype)

    def norm(self, i, ord=None, dim=None):
        return np.linalg.norm(i, ord=ord, axis=dim)

    def argsort(self, ipt, dim=-1, **kwargs):
        kind = kwargs.pop('kind', None)
        order = kwargs.pop('order', None)
        return np.argsort(ipt, dim, kind=kind, order=order)

    def sort(self, i, dim=-1, **kwargs):
        order = kwargs.pop('order', None)
        kind = kwargs.pop('kind', 'quicksort')
        return np.sort(i, axis=dim, kind=kind, order=order)

    def delete(self, x, idx, axis=None):
        return np.delete(x, idx, axis=axis)

    def block(self, arr):
        return np.block(arr)

    def isnan(self, a):
        return np.isnan(a)

    @staticmethod
    def prec_fix(arr, prec=15):
        # Calculate the scaling factor
        # For real numbers
        scale = 10 ** prec
        if np.isrealobj(arr):
            return np.floor(arr * scale) / scale

        # For complex numbers
        else:
            real = np.floor(arr.real * scale) / scale
            imag = np.floor(arr.imag * scale) / scale
            return real + 1j * imag

    def sub(self, a, b):
        return self.prec_fix(a - b)

    def div(self, a, b):
        return self.prec_fix(a / b)

    def get_gradient(self, y, x):
        warnings.warn("Auto differentiation is not available on Numpy, "
                      "this behavior will be the same as np.gradient. "
                      "Recommend 30 samples from range [x - 0.1, x + 0.1] "
                      "to get a relatively accurate gradient.")
        """
        The gradient is computed using second order accurate central differences 
        in the interior points and either first or second order accurate one-sides
         (forward or backwards) differences at the boundaries.
        """
        if type(y) is not list:
            y = [y]

        if type(x) is not list:
            x = [x]

        if len(y) != len(x) or len(y) < 2:
            print("You need a series of y and x values for gradient inference.")
            return
        return np.gradient(y,x)