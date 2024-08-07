from warnings import warn

import numpy as np
import scipy.linalg as sla
import scipy.fft as sfft
from inkstone.backends.GenericBackend import GenericBackend

class NumpyBackend(GenericBackend):

    def __init__(self):

        super().__init__()
        self.raw_type = np.ndarray
        self.abs = np.abs
        self.sqrt = np.sqrt
        self.arange = np.arange
        self.ceil = np.ceil
        self.where = np.where
        self.la = np.linalg
        self.diag = np.diag
        self.sin = np.sin
        self.cos = np.cos
        self.arccos = np.arccos
        self.arcsin = np.arcsin
        self.ones = np.ones
        self.square = np.square
        self.concatenate = np.concatenate
        self.conj = np.conj
        self.exp = np.exp
        self.sinc = np.sinc
        self.zeros = np.zeros
        self.tan = np.tan
        self.roll = np.roll
        self.sum = np.sum
        self.dot = np.dot
        self.hsplit = np.hsplit
        self.repeat = np.repeat
        self.reshape = np.reshape
        self.moveaxis = np.moveaxis
        self.full = np.full
        self.logical_not = np.logical_not
        self.maximum = np.maximum
        self.einsum = np.einsum
        self.lu_factor = sla.lu_factor
        self.fft = sfft
        self.solve = sla.solve
        self.linspace = np.linspace
        self.cross = np.cross

        self.slogdet = np.linalg.slogdet

        self.pi = np.pi
        self.float64 = np.float64
        self.int32 = np.int32
        self.complex128 = np.complex128
        self.eye = np.eye


    def parseData(self, i: any, dtype=None, **kwargs):
        return np.array(i, dtype=dtype)

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

    def triu_indices(self, row, col=None, offset=0):
        if not col:
            col = row
        return np.triu_indices(row, offset, col)

    def lu_solve(self, p, q):
        return sla.lu_solve(p, q)

    def norm(self, i, dim=None):
        return np.linalg.norm(i, axis=dim)

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