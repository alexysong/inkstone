import numpy as np
import jax.numpy as jnp

from inkstone_old_package import Inkstone as OldInkstone

from inkstone import GenericBackend
from inkstone.simulator import Inkstone as NewInkstone

def one_layer_simulation_1D(frequency=0.4, theta=0., phi=0., 
                            lattice=1., slab_thickness=0.55, slab_permittivity=12, resonator_width=0.45, resonator_center=0.5, 
                            num_g=30, version="old", backend="numpy"):
    if version == "old":
        s = OldInkstone()
    elif version == "new":
        GenericBackend.switchTo(backend)
        s = NewInkstone()
    else:
        print("version string not recognised. Use 'old' for v0.3.10 and 'new' for new backends.") 
        s = OldInkstone()
    
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

abs_tol = 1e-6
rel_tol = 1e-6

simulation_0310 = one_layer_simulation_1D()
fields = simulation_0310.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)

simulation_nb = one_layer_simulation_1D(version="new", backend="numpy")
fields_nb = simulation_nb.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)

simulation_jb = one_layer_simulation_1D(version="new", backend="jax")
fields_jb = simulation_jb.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)

for f, f_nb, f_jb in zip(fields,fields_nb,fields_jb):
    assert np.allclose(f_nb, f, rtol=rel_tol, atol=abs_tol) and np.allclose(f_jb, f, rtol=rel_tol, atol=abs_tol)