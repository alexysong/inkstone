import pytest

import numpy as np
import jax.numpy as jnp

from inkstone_old_package import Inkstone as OldInkstone

from ... import GenericBackend
from ...simulator import Inkstone as NewInkstone



# SETTINGS ################################################################################################
np.random.seed(2024)
abs_tol = 1e-10
rel_tol = 1e-10



# SETUP ################################################################################################
@pytest.fixture
def one_layer_simulation_1D():
    def _one_layer_simulation_1D(frequency=0.4, theta=0., phi=0., 
                                lattice=1., slab_thickness=0.55, slab_permittivity=12, resonator_width=0.45, resonator_center=0.5, 
                                num_g=30, version="old", backend="numpy"):
        if version == "old":
            s = OldInkstone()
        elif version == "new":
            GenericBackend.switchTo(backend)
            # GenericBackend.genericBackend = GenericBackend.GenericBackend(backend)
            s = NewInkstone()
        else:
            raise ValueError("version string not recognised. Use 'old' for v0.3.10 and 'new' for new backends.") 
        
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

    return _one_layer_simulation_1D

@pytest.fixture
def one_layer_simulation_2D():
    def _one_layer_simulation_2D(frequency=0.4, theta=0., phi=0., 
                                lattice=((1, 0), (0, 1)), slab_thickness=0.55, slab_permittivity=12, 
                                pattern_radius=0.2, pattern_center=0.45,
                                num_g=100, version="old", backend="numpy"):
        if version == "old":
            s = OldInkstone()
        elif version == "new":
            GenericBackend.switchTo(backend)
            # GenericBackend.genericBackend = GenericBackend.GenericBackend(backend)
            s = NewInkstone()
        else:
            raise ValueError("version string not recognised. Use 'old' for v0.3.10 and 'new' for new backends.") 
        
        s.lattice = lattice
        s.num_g = num_g
        s.frequency = frequency

        s.AddMaterial(name='di', epsilon=slab_permittivity)

        s.AddLayer(name='in', thickness=0, material_background='vacuum')
        s.AddLayer(name='slab', thickness=slab_thickness, material_background='di')
        s.AddLayerCopy(name='out', original_layer='in', thickness=0)


        s.AddPatternDisk(layer='slab', pattern_name='disk', material='vacuum', radius=0.2)

        # Incident wave
        s.SetExcitation(theta=theta, phi=phi, s_amplitude=1, p_amplitude=0)

        return s

    return _one_layer_simulation_2D



# 1D SIMULATION ################################################################################################
def test_one_layer_fields_1D(one_layer_simulation_1D):
    """
    Test one_layer_simulation_1D for all backends is consistent with v0.3.10's numpy 
    """
    simulation_0310 = one_layer_simulation_1D(version="old", backend="numpy")
    fields = simulation_0310.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)
    
    simulation_nb = one_layer_simulation_1D(version="new", backend="numpy")
    fields_nb = simulation_nb.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)
    
    simulation_jb = one_layer_simulation_1D(version="new", backend="jax")
    fields_jb = simulation_jb.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)
    
    for f, f_nb, f_jb in zip(fields,fields_nb,fields_jb):
        assert np.allclose(f_nb, f, rtol=rel_tol, atol=abs_tol) and np.allclose(f_jb, f, rtol=rel_tol, atol=abs_tol)

def test_one_layer_efficiencies_1D(one_layer_simulation_1D):
    """
    Test one_layer_simulation_1D for all backends is consistent with v0.3.10's numpy 
    """
    simulation_0310 = one_layer_simulation_1D(version="old", backend="numpy")
    i, r = simulation_0310.GetPowerFlux('in')
    t = simulation_0310.GetPowerFlux('out')[0] / i
    efficencies = [i,r,t]
    
    simulation_nb = one_layer_simulation_1D(version="new", backend="numpy")
    i_nb, r_nb = simulation_nb.GetPowerFlux('in')
    t_nb = simulation_nb.GetPowerFlux('out')[0] / i_nb
    efficencies_nb = [i_nb,r_nb,t_nb]
    
    simulation_jb = one_layer_simulation_1D(version="new", backend="jax")
    i_jb, r_jb = simulation_jb.GetPowerFlux('in')
    t_jb = simulation_jb.GetPowerFlux('out')[0] / i_jb
    efficencies_jb = [i_jb,r_jb,t_jb]
    
    for e, e_nb, e_jb in zip(efficencies,efficencies_nb,efficencies_jb):
        assert np.allclose(e_nb, e, rtol=rel_tol, atol=abs_tol) and np.allclose(e_jb, e, rtol=rel_tol, atol=abs_tol)



# 2D SIMULATION ################################################################################################
def test_one_layer_fields_2D(one_layer_simulation_2D):
    """
    Test one_layer_simulation_1D for all backends is consistent with v0.3.10's numpy 
    """
    simulation_0310 = one_layer_simulation_2D(version="old", backend="numpy")
    fields = simulation_0310.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)
    
    simulation_nb = one_layer_simulation_2D(version="new", backend="numpy")
    fields_nb = simulation_nb.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)
    
    simulation_jb = one_layer_simulation_2D(version="new", backend="jax")
    fields_jb = simulation_jb.GetFields(xmin=-0.5, xmax=0.5, nx=101, y=0, zmin=-0.2, zmax=0.75, nz=101)
    
    for f, f_nb, f_jb in zip(fields,fields_nb,fields_jb):
        assert np.allclose(f_nb, f, rtol=rel_tol, atol=abs_tol) and np.allclose(f_jb, f, rtol=rel_tol, atol=abs_tol)

def test_one_layer_efficencies_2D(one_layer_simulation_2D):
    """
    Test one_layer_simulation_1D for all backends is consistent with v0.3.10's numpy 
    """
    simulation_0310 = one_layer_simulation_2D(version="old", backend="numpy")
    i, r = simulation_0310.GetPowerFlux('in')
    t = simulation_0310.GetPowerFlux('out')[0] / i
    efficencies = [i,r,t]
    
    simulation_nb = one_layer_simulation_2D(version="new", backend="numpy")
    i_nb, r_nb = simulation_nb.GetPowerFlux('in')
    t_nb = simulation_nb.GetPowerFlux('out')[0] / i_nb
    efficencies_nb = [i_nb,r_nb,t_nb]
    
    simulation_jb = one_layer_simulation_2D(version="new", backend="jax")
    i_jb, r_jb = simulation_jb.GetPowerFlux('in')
    t_jb = simulation_jb.GetPowerFlux('out')[0] / i_jb
    efficencies_jb = [i_jb,r_jb,t_jb]
    
    for e, e_nb, e_jb in zip(efficencies,efficencies_nb,efficencies_jb):
        assert np.allclose(e_nb, e, rtol=rel_tol, atol=abs_tol) and np.allclose(e_jb, e, rtol=rel_tol, atol=abs_tol)