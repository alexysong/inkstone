import numpy as np
import jax.numpy as jnp
from jax import grad as jax_grad

from inkstone import GenericBackend
from inkstone.simulator import Inkstone as NewInkstone # rename to NewInkstone so switchTo works correctly

def reflection_1d(d,w,f,permittivity,num_g=30,backend="jax"):
    GenericBackend.switchTo(backend)
    s = NewInkstone()
    s.frequency = f
    s.lattice = 1
    s.num_g = num_g

    s.AddMaterial(name='di', epsilon=permittivity)

    s.AddLayer(name='in', thickness=0, material_background='vacuum')
    s.AddLayer(name='slab', thickness=d, material_background='di')
    s.AddLayerCopy(name='out', original_layer='in', thickness=0)

    s.AddPattern1D(layer='slab', pattern_name='box', material='vacuum', width=w, center=0.5)

    s.SetExcitation(theta=0, phi=0, s_amplitude=1, p_amplitude=0)

    i, r = s.GetPowerFlux('in')
    return -r / i

eps = 1e-7
abs_tol = 0
rel_tol = 1e-6

params_1d = (1.0, 0.4, 0.2, 12.)
grating_widths = jnp.linspace(0.01,0.99,5, dtype=jnp.float64)
w = grating_widths[0]
reflection_1d_grad0 = jax_grad(reflection_1d,0)
reflection_1d_grad1 = jax_grad(reflection_1d,1)


# for w in grating_widths:
d = params_1d[0]

params = params_1d[1:]
FD_grad = (reflection_1d(d+eps/2,*params,num_g=30,backend="jax") - reflection_1d(d-eps/2,*params,num_g=30,backend="jax"))/eps
AD_grad = reflection_1d_grad0(d,*params,backend="jax")

params = params_1d[2:]
# FD_grad = (reflection_1d(d,w+eps/2,*params,num_g=30,backend="jax") - reflection_1d(d,w-eps/2,*params,num_g=30,backend="jax"))/eps
# AD_grad = reflection_1d_grad1(d,w,*params,num_g=30,backend="jax")

assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)