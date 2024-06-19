from functools import partial
import numpy.linalg as npla
import numpy  as anp
from autograd.extend import defvjp,primitive

# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

# transpose by swapping last two dimensions
def T(x): return anp.swapaxes(x, -1, -2)

_dot = partial(anp.einsum, '...ij,...jk->...ik')

inv = primitive(anp.linalg.inv)
def grad_inv(ans, x):
    return lambda g: -_dot(_dot(T(ans), g), T(ans))
defvjp(inv, grad_inv)


