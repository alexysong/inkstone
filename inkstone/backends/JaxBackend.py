# TODO: all numbers should be torch.tensor

import jax
import jaxlib
from inkstone.primitives.jax_primitive import j0, j1, eig
from inkstone.backends.GenericBackend import GenericBackend

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp


class JaxBackend(GenericBackend):
    def __init__(self):
        super().__init__()
        self.raw_type = jaxlib.xla_extension.ArrayImpl

        self.abs = jnp.abs
        self.arange = jnp.arange
        self.arccos = jnp.arccos
        self.arcsin = jnp.arcsin
        self.ceil = jnp.ceil
        self.concatenate = jnp.concatenate
        self.conj = jnp.conj
        self.cos = jnp.cos
        self.diag = jnp.diag
        self.dot = jnp.dot
        self.einsum = jnp.einsum
        self.exp = jnp.exp
        self.eye = jnp.eye
        self.fft = jnp.fft
        self.full = jnp.full
        self.hsplit = jnp.hsplit
        self.isnan = jnp.isnan
        self.la = jnp.linalg
        self.linspace = jnp.linspace
        self.logspace = jnp.logspace
        self.logical_not = jnp.logical_not
        self.lu_factor = jsp.linalg.lu_factor
        self.maximum = jnp.maximum
        self.moveaxis = jnp.moveaxis
        self.repeat = jnp.repeat
        self.reshape = jnp.reshape
        self.roll = jnp.roll
        self.rollaxis = jnp.rollaxis
        self.sin = jnp.sin
        self.sinc = jnp.sinc
        self.sla = jsp.linalg
        self.stack = jnp.stack
        self.solve = jsp.linalg.solve
        self.sqrt = jnp.sqrt
        self.square = jnp.square
        self.sum = jnp.sum
        self.tan = jnp.tan
        self.where = jnp.where

        self.j0 = j0
        self.j1 = j1
        self.eig = eig

        self.complex128 = jnp.complex128
        self.float64 = jnp.float64
        self.int32 = jnp.int32
        self.pi = jnp.pi

    def data(self, i: any, dtype=None, **kwargs):
        #if isinstance(i, jax.Array):  # handle tracer inputs by not passing invalid dtype
        """
        JL
        TODO:
        Need more robust handling of list of tracer inputs to gb.data
        Currently, if o is a JAX tracer, then the above control flow sets dtype = jax tracer.
        Since JAX does not accept dtype = jax tracer in jnp.array(), it throws an error.
        Workaround is manually setting the data argument dtype whenever JAX throws an error
        to avoid setting dtype = type(o)
        Potential fix is to manually check isinstance(o, jax.Array), but that requires the user
        to have installed JAX, which they may not have if they only want to use one of the other backends.
        Need to somehow detect innermost o type as tracer and set jnp.array(i, dtype=None) without
        re-calculating “o” or calculating “o” in cases where it is not needed

        This isinstance(i, jax.Array) check loses dtype from the previous control flow, is that
        acceptable? The dtype is then implicitly set by the dtype of o, i.e. the dtype in the tracer arrays.
        """
        return jnp.array(i)

    def isnan(self, a):
        return jnp.isnan(a)

    def zeros(self, a, dtype=jnp.float64):# only numpy fft used
        return jnp.zeros(a, dtype=dtype)

    def ones(self, a, dtype=jnp.float64):
        return jnp.ones(a, dtype=dtype)

    def parseList(self, tup):
        return jnp.array(tup)

    def meshgrid(self, *xi):
        return jnp.meshgrid(*xi)

    def castType(self, i, typ):  # type => typ, avoid collision with keyword
        return i.astype(typ)

    def cross(self, a, b, dim=None):
        return jnp.cross(a, b, axis=dim)

    def getSize(self, i):
        return i.size

    def delete(self, x, idx, axis=None):
        return jnp.delete(x, idx, axis=axis)

    def clone(self, i, keep_grad=False):
        return i

    def triu_indices(self, row, col=None, offset=0):
        if not col:
            col = row
        return jnp.triu_indices(row, offset, col)

    def argsort(self, a, dim=-1, **kwargs):
        kind = kwargs.pop('kind',None)
        order= kwargs.pop('order',None)
        stable = kwargs.pop('stable',True)
        return jnp.argsort(a, axis=dim, kind=kind, order=order, stable=stable)

    def sort(self, i, dim=-1, **kwargs):
        des = kwargs.pop('des',False)
        kind = kwargs.pop('kind', None)
        order = kwargs.pop('order', None)
        return jnp.sort(i, axis=dim, descending=des,kind=kind,order=order)

    #  def partition(self, i, kth, dim=-1):
    #       match self.backend:
    #         case "torch":
    #               return torch.topk(i,kth,dim)
    #            case "autograd":
    #              return anp.partition(i,kth,dim)
    #            case "jax":
    #              return jnp.partition(i,kth,dim)
    #            case "numpy":
    #              return np.partition(i,kth,dim)

    def block(self, arr):
        return jnp.block(arr)

    def lu_solve(self, p, q):
        return jsp.linalg.lu_solve(p, q)

    def norm(self, p, ord=None, dim=None):
        return jnp.linalg.norm(p,ord=ord,axis=dim)

    def indexAssign(self, a, idx, b):
        """
        For numpy, use index assignment. For differentiation libraries, replace with differentiable version
        """
        return a.at[idx].set(b)

    def assignMul(self, a, idx, b):
        """
        For numpy, multiply in-place with index assignment. For differentiation libraries, replace with differentiable not-in-place version
        """
        return a.at[idx].multiply(b)
