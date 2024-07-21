import autograd.numpy as anp
import numpy as np
import scipy.linalg as sla

from inkstone.backends.GenericBackend import GenericBackend


class AutogradBackend(GenericBackend):

    def __init__(self):
        super().__init__()
        self.raw_type = anp.ndarray
        self.abs = anp.abs
        self.sqrt = anp.sqrt
        self.arange = anp.arange
        self.ceil = anp.ceil
        self.where = anp.where
        self.la = anp.linalg
        self.lu_factor = sla.lu_factor
        self.diag = anp.diag
        self.sin = anp.sin
        self.cos = anp.cos
        self.arccos = anp.arccos
        self.arcsin = anp.arcsin
        self.ones = anp.ones
        self.square = anp.square
        self.concatenate = anp.concatenate
        self.concatenate = anp.concatenate
        self.conj = anp.conj
        self.exp = anp.exp
        self.sinc = anp.sinc
        self.zeros = anp.zeros
        self.tan = anp.tan
        self.roll = anp.roll
        self.sum = anp.sum
        self.dot = anp.dot
        self.hsplit = anp.hsplit
        self.repeat = anp.repeat
        self.reshape = anp.reshape
        self.moveaxis = anp.moveaxis
        self.full = anp.full
        self.logical_not = anp.logical_not
        self.maximum = anp.maximum
        self.einsum = anp.einsum
        self.linspace = anp.linspace
        self.fft = anp.fft
        self.solve = sla.solve

        self.pi = np.pi
        self.float64 = np.float64
        self.int32 = np.int32
        self.complex128 = np.complex128
        self.eye = anp.eye


    def parseData(self, i: any, dtype=None):
        return anp.array(i, dtype=dtype)

    def meshgrid(self, a, b):
        return anp.meshgrid(a, b)


    def castType(self, i, typ):  # typ(e), avoid collision with keyword
        return i.astype(typ)

    def cross(self, a, b):
        return anp.cross(a, b)


    def laCross(self, a1, a2):
        return anp.linalg.cross(a1, a2)


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
