"""
Debugging backend changing methods to ensure complete backend switch

"""

import numpy as np
import jax.numpy as jnp

from .. import GenericBackend
from ..simulator import Inkstone


# SETTINGS ################################################################################################
np.random.seed(2024)
abs_tol = 1e-8



# SETUP ################################################################################################
def debug_one_layer_simulation_1D(frequency=0.4, theta=0., phi=0., 
                                lattice=1., slab_thickness=0.55, slab_permittivity=12, resonator_width=0.45, resonator_center=0.5, 
                                num_g=30, version="old", backend="numpy"):
    if version == "old":
        raise NotImplementedError("Need to determine how to import inkstone v0.3.10 in test file") 
    elif version == "new":
        GenericBackend.switchTo(backend)
        # GenericBackend.genericBackend = GenericBackend.GenericBackend(backend)
        s = Inkstone()
    else:
        print("version string not recognised. Use 'old' for v0.3.10 and 'new' for new backends.") 
        s = Inkstone()
    
    s.lattice = lattice
    s.num_g = num_g
    s.frequency = frequency

    s.AddMaterial(name='di', epsilon=slab_permittivity)

    s.AddLayer(name='in', thickness=0, material_background='vacuum')
    s.AddLayer(name='slab', thickness=slab_thickness, material_background='di')
    s.AddLayerCopy(name='out', original_layer='in', thickness=0)

    s.AddPattern1D(layer='slab', pattern_name='grating', material='vacuum', 
                width=resonator_width, center=resonator_center)

    # Incident wave
    s.SetExcitation(theta=theta, phi=phi, s_amplitude=1, p_amplitude=0)

    return s

# debug_one_layer_simulation_1D(version="new", backend="numpy")
# s = debug_one_layer_simulation_1D(version="new", backend="jax")
simulation_nb = debug_one_layer_simulation_1D(version="new", backend="numpy")
fields_nb = simulation_nb.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)
simulation_jb = debug_one_layer_simulation_1D(version="new", backend="jax")
fields_jb = simulation_jb.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)

for f_nb, f_jb in zip(fields_nb,fields_jb):
    if (f_nb - f_jb < abs_tol).all():
        pass
    else:
        assert False