import autograd.numpy as anp
from autograd import grad
from inkstone.backends.primitives.autograd_primitive import j0,j1,eig
import numpy as np
import scipy.linalg as sla

from inkstone.backends.Backend import Backend


class AutogradBackend(Backend):

    def __init__(self):
        super().__init__()
        self.raw_type = anp.ndarray
        self.ifftshift = anp.fft.ifftshift

        self.pi = np.pi
        self.float64 = np.float64
        self.int32 = np.int32
        self.complex128 = np.complex128

    def j0(self, *args, **kwargs):
        return j0(*args, **kwargs)

    def j1(self, *args, **kwargs):
        return j1(*args, **kwargs)

    def eig(self, *args, **kwargs):
        return eig(*args, **kwargs)

    def data(self, i: any, dtype=None, **kwargs):
        return anp.array(i, dtype=dtype)

    def meshgrid(self, *xi):
        return anp.meshgrid(*xi)

    def zeros(self, a, dtype):
        return anp.zeros(a, dtype=dtype)

    def ones(self, a, dtype):
        return anp.ones(a, dtype=dtype)

    def castType(self, i, typ):  # typ(e), avoid collision with keyword
        return i.astype(typ)

    def cross(self, a, b):
        return anp.cross(a, b)

    def getSize(self, i):
        return i.size

    def delete(self, x, idx, axis=None):
        return anp.delete(x, idx, axis=axis)

    def clone(self, i, keep_grad=False):
        return anp.copy(i, order='C', subok=True)

    def triu_indices(self, row, col=None, offset=0):
        if not col:
            col = row
        return anp.triu_indices(row, offset, col)

    def argsort(self, a, b=-1, c=None, d=None):
        return anp.argsort(a, b, c, d)

    def parseList(self, tup):
        return anp.array(tup)

    def lu_solve(self, p, q):
        return sla.lu_solve(p, q)

    def norm(self, a, dim=None):
        return anp.linalg.norm(a, axis=dim)

    def sort(self, i, dim=-1, des=False, sort_alg='quicksort'):
        return anp.sort(i, dim, sort_alg)

    def block(self, arr):
        return anp.block(arr)

    def isnan(self, a):
        return anp.isnan(a)

    def abs(self, *args, **kwargs):
        return anp.abs(*args, **kwargs)

    def sqrt(self, *args, **kwargs):
        return anp.sqrt(*args, **kwargs)

    def arange(self, *args, **kwargs):
        return anp.arange(*args, **kwargs)

    def ceil(self, *args, **kwargs):
        return anp.ceil(*args, **kwargs)

    def where(self, *args, **kwargs):
        return anp.where(*args, **kwargs)

    def lu_factor(self, *args, **kwargs):
        return sla.lu_factor(*args, **kwargs)

    def diag(self, *args, **kwargs):
        return anp.diag(*args, **kwargs)

    def sin(self, *args, **kwargs):
        return anp.sin(*args, **kwargs)

    def cos(self, *args, **kwargs):
        return anp.cos(*args, **kwargs)

    def arccos(self, *args, **kwargs):
        return anp.arccos(*args, **kwargs)

    def arcsin(self, *args, **kwargs):
        return anp.arcsin(*args, **kwargs)

    def ones(self, *args, **kwargs):
        return anp.ones(*args, **kwargs)

    def square(self, *args, **kwargs):
        return anp.square(*args, **kwargs)

    def stack(self, *args, **kwargs):
        return anp.stack(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        return anp.concatenate(*args, **kwargs)

    def conj(self, *args, **kwargs):
        return anp.conj(*args, **kwargs)

    def exp(self, *args, **kwargs):
        return anp.exp(*args, **kwargs)

    def sinc(self, *args, **kwargs):
        return anp.sinc(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return anp.zeros(*args, **kwargs)

    def tan(self, *args, **kwargs):
        return anp.tan(*args, **kwargs)

    def roll(self, *args, **kwargs):
        return anp.roll(*args, **kwargs)

    def sum(self, *args, **kwargs):
        return anp.sum(*args, **kwargs)

    def dot(self, *args, **kwargs):
        return anp.dot(*args, **kwargs)

    def hsplit(self, *args, **kwargs):
        return anp.hsplit(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        return anp.repeat(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return anp.reshape(*args, **kwargs)

    def moveaxis(self, *args, **kwargs):
        return anp.moveaxis(*args, **kwargs)

    def full(self, *args, **kwargs):
        return anp.full(*args, **kwargs)

    def logical_not(self, *args, **kwargs):
        return anp.logical_not(*args, **kwargs)

    def maximum(self, *args, **kwargs):
        return anp.maximum(*args, **kwargs)

    def einsum(self, *args, **kwargs):
        return anp.einsum(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return anp.linspace(*args, **kwargs)

    def solve(self, *args, **kwargs):
        return sla.solve(*args, **kwargs)

    def eye(self, *args, **kwargs):
        return anp.eye(*args, **kwargs)

    def slogdet(self, *args, **kwargs):
        return anp.linalg.slogdet(*args, **kwargs)

    def get_gradient(self, y, x):
        y_grad = grad(y)
        return y_grad(x)
