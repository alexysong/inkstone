import numpy as np

import jax
import jax.numpy as jnp

from ... import GenericBackend
from ...simulator import Inkstone

def reflection_1d(d,w,f,permittivity,num_g=30,backend="jax"):
    GenericBackend.switchTo(backend)
    s = Inkstone()
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