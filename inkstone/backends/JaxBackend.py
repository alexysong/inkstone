# TODO: all numbers should be torch.tensor

import jax
import jaxlib
from inkstone.primitives.primitives import j0, j1, eig
from inkstone.backends.GenericBackend import GenericBackend

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp


class JaxBackend(GenericBackend):
    def __init__(self):
        super().__init__()
        self.raw_type = jaxlib.xla_extension.ArrayImpl

        self.abs = jnp.abs
        self.sqrt = jnp.sqrt
        self.arange = jnp.arange
        self.ceil = jnp.ceil
        self.where = jnp.where
        self.diag = jnp.diag
        self.sin = jnp.sin
        self.cos = jnp.cos
        self.arccos = jnp.arccos
        self.arcsin = jnp.arcsin
        self.ones = jnp.ones
        self.square = jnp.square
        self.concatenate = jnp.concatenate
        self.conj = jnp.conj
        self.exp = jnp.exp
        self.sinc = jnp.sinc
        self.zeros = jnp.zeros
        self.tan = jnp.tan
        self.roll = jnp.roll
        self.sum = jnp.sum
        self.dot = jnp.dot
        self.hsplit = jnp.hsplit
        self.repeat = jnp.repeat
        self.reshape = jnp.reshape
        self.rollaxis = jnp.rollaxis
        self.moveaxis = jnp.moveaxis
        self.full = jnp.full
        self.logical_not = jnp.logical_not
        self.maximum = jnp.maximum
        self.einsum = jnp.einsum
        self.isnan = jnp.isnan

        self.la = jnp.linalg
        self.sla = jsp.linalg
        self.fft = jnp.fft  # only numpy fft used

        self.j0 = j0
        self.j1 = j1
        self.eig = eig

        self.pi = jnp.pi
        self.float64 = jnp.float64
        self.int32 = jnp.int32
        self.complex128 = jnp.complex128
        self.eye = jnp.eye


    def parseData(self, i: any, dtype=None):

        if isinstance(i, jax.Array):  # handle tracer inputs by not passing invalid dtype
            """
            JL
            TODO:
            Need more robust handling of list of tracer inputs to self.gb.parseData
            Currently, if o is a JAX tracer, then the above control flow sets dtype = jax tracer.
            Since JAX does not accept dtype = jax tracer in jnp.array(), it throws an error.
            Workaround is manually setting the parseData argument dtype whenever JAX throws an error 
            to avoid setting dtype = type(o)
            Potential fix is to manually check isinstance(o, jax.Array), but that requires the user 
            to have installed JAX, which they may not have if they only want to use one of the other backends.
            Need to somehow detect innermost o type as tracer and set jnp.array(i, dtype=None) without 
            re-calculating “o” or calculating “o” in cases where it is not needed

            This isinstance(i, jax.Array) check loses dtype from the previous control flow, is that
            acceptable? The dtype is then implicitly set by the dtype of o, i.e. the dtype in the tracer arrays.
            """
            return jnp.array(i)

    def meshgrid(self, a, b):
        return jnp.meshgrid(a, b)


    def castType(self, i, typ):  # typ(e), avoid collision with keyword
        return i.astype(typ)

    def cross(self, a, b):
        return jnp.cross(a, b)

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

    def argsort(self, a, b=-1, c=None, d=None):
        return jnp.argsort(a, b, c, d)


    def sort(self, i, dim=-1, des=False, sort_alg='quicksort'):
        return jnp.sort(i,axis=dim,descending=des)


    def linspace(self, start, end, num=50, required_grad=False):
        return jnp.linspace(start, end, num)

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

    def indexAssign(self, a, idx, b):
        """
        For numpy, use index assignment. For differentiation libraries, replace with differentiable version
        """
        return a.at[idx].set(b)

    def assignAndMultiply(self, a, idx, b):
        """
        For numpy, multiply in-place with index assignment. For differentiation libraries, replace with differentiable not-in-place version
        """
        return a.at[idx].multiply(b)

