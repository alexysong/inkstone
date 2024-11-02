# TODO: all numbers should be torch.tensor

import jax
import jaxlib
from inkstone.backends.primitives.jax_primitive import j0, j1, eig
from inkstone.backends.Backend import Backend

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp


class JaxBackend(Backend):
    def __init__(self):
        super().__init__()
        self.raw_type = jaxlib.xla_extension.ArrayImpl

        self.ifftshift = jnp.fft.ifftshift

        self.complex128 = jnp.complex128
        self.float64 = jnp.float64
        self.int32 = jnp.int32
        self.pi = jnp.pi

    def abs(self, *args, **kwargs):
        return jnp.abs(*args, **kwargs)

    def arange(self, *args, **kwargs):
        return jnp.arange(*args, **kwargs)

    def arccos(self, *args, **kwargs):
        return jnp.arccos(*args, **kwargs)

    def arcsin(self, *args, **kwargs):
        return jnp.arcsin(*args, **kwargs)

    def ceil(self, *args, **kwargs):
        return jnp.ceil(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        return jnp.concatenate(*args, **kwargs)

    def conj(self, *args, **kwargs):
        return jnp.conj(*args, **kwargs)

    def cos(self, *args, **kwargs):
        return jnp.cos(*args, **kwargs)

    def diag(self, *args, **kwargs):
        return jnp.diag(*args, **kwargs)

    def dot(self, *args, **kwargs):
        return jnp.dot(*args, **kwargs)

    def einsum(self, *args, **kwargs):
        return jnp.einsum(*args, **kwargs)

    def exp(self, *args, **kwargs):
        return jnp.exp(*args, **kwargs)

    def eye(self, *args, **kwargs):
        return jnp.eye(*args, **kwargs)

    def full(self, *args, **kwargs):
        return jnp.full(*args, **kwargs)

    def hsplit(self, *args, **kwargs):
        return jnp.hsplit(*args, **kwargs)

    def isnan(self, *args, **kwargs):
        return jnp.isnan(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return jnp.linspace(*args, **kwargs)

    def logspace(self, *args, **kwargs):
        return jnp.logspace(*args, **kwargs)

    def logical_not(self, *args, **kwargs):
        return jnp.logical_not(*args, **kwargs)

    def lu_factor(self, *args, **kwargs):
        return jsp.linalg.lu_factor(*args, **kwargs)

    def maximum(self, *args, **kwargs):
        return jnp.maximum(*args, **kwargs)

    def moveaxis(self, *args, **kwargs):
        return jnp.moveaxis(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        return jnp.repeat(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return jnp.reshape(*args, **kwargs)

    def roll(self, *args, **kwargs):
        return jnp.roll(*args, **kwargs)

    def rollaxis(self, *args, **kwargs):
        return jnp.rollaxis(*args, **kwargs)

    def sin(self, *args, **kwargs):
        return jnp.sin(*args, **kwargs)

    def sinc(self, *args, **kwargs):
        return jnp.sinc(*args, **kwargs)

    def stack(self, *args, **kwargs):
        return jnp.stack(*args, **kwargs)


    def slogdet(self, *args, **kwargs):
        return jnp.linalg.slogdet(*args, **kwargs)
    def solve(self, *args, **kwargs):
        return jsp.linalg.solve(*args, **kwargs)

    def sqrt(self, *args, **kwargs):
        return jnp.sqrt(*args, **kwargs)

    def square(self, *args, **kwargs):
        return jnp.square(*args, **kwargs)

    def sum(self, *args, **kwargs):
        return jnp.sum(*args, **kwargs)

    def tan(self, *args, **kwargs):
        return jnp.tan(*args, **kwargs)

    def where(self, *args, **kwargs):
        return jnp.where(*args, **kwargs)

    def j0(self, *args, **kwargs):
        return j0(*args, **kwargs)

    def j1(self, *args, **kwargs):
        return j1(*args, **kwargs)

    def eig(self, *args, **kwargs):
        return eig(*args, **kwargs)

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

    def get_gradient(self, y, x):
        """

        Parameters
        ----------
        y   : the calculation function (not the result value)
        x   : the parameter with respect to y

        Returns
        -------

        """
        grad_loss = jax.jit(jax.grad(y))
        return grad_loss(x)
