from warnings import warn

import numpy as np
import scipy.linalg as sla
import scipy.special as sp
import scipy.fft as sfft
from inkstone.backends.GenericBackend import GenericBackend

class NumpyBackend(GenericBackend):

    def __init__(self):

        super().__init__()
        self.raw_type = np.ndarray

        self.abs = np.abs
        self.arange = np.arange
        self.arccos = np.arccos
        self.arcsin = np.arcsin
        self.ceil = np.ceil
        self.concatenate = np.concatenate
        self.conj = np.conj
        self.cos = np.cos
        self.diag = np.diag
        self.dot = np.dot
        self.einsum = np.einsum
        self.exp = np.exp
        self.eye = np.eye
        self.fft = sfft
        self.full = np.full
        self.hsplit = np.hsplit
        self.la = np.linalg
        self.linspace = np.linspace
        self.logical_not = np.logical_not
        self.lu_factor = sla.lu_factor
        self.maximum = np.maximum
        self.moveaxis = np.moveaxis
        self.ones = np.ones
        self.repeat = np.repeat
        self.reshape = np.reshape
        self.roll = np.roll
        self.sin = np.sin
        self.sinc = np.sinc
        self.slogdet = np.linalg.slogdet
        self.solve = sla.solve
        self.square = np.square
        self.sqrt = np.sqrt
        self.sum = np.sum
        self.tan = np.tan
        self.where = np.where
        self.zeros = np.zeros
        self.stack = np.stack

        self.j0 = sp.j0
        self.j1 = sp.j1

        self.complex128 = np.complex128
        self.float64 = np.float64
        self.int32 = np.int32
        self.pi = np.pi

    @staticmethod
    def fix_eigenvector_phase(x):
        t = np.sqrt(x)
        return np.where(t.imag<0,-t,t)

    def eig(self, A):
        r = sla.eig(A)
        return self.prec_fix(r[0]), self.prec_fix(r[1])

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

    def ones(self,p,dtype=np.float64):
        return np.ones(p,dtype=dtype)

    def zeros(self, p ,dtype=np.float64):
        return np.zeros(p,dtype=dtype)
    def norm(self, i, ord=None, dim=None):
        return np.linalg.norm(i, ord=ord, axis=dim)

    def argsort(self, ipt, dim=-1, **kwargs):
        kind = kwargs.pop('kind',None)
        order = kwargs.pop('order',None)
        return np.argsort(ipt,dim,kind=kind,order=order)

    def sort(self,i,dim=-1,**kwargs):
        order = kwargs.pop('order',None)
        kind = kwargs.pop('kind','quicksort')
        return np.sort(i,axis=dim,kind=kind,order=order)

    def delete(self, x, idx, axis=None):
        return np.delete(x,idx, axis=axis)

    def block(self,arr):
        return np.block(arr)

    def isnan(self,a):
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