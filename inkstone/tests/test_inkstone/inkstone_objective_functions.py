import numpy as np

from ... import GenericBackend 
from ...simulator import Inkstone

## 1 layer 1d ##
def simulation_1layer_1d(d,w,f,p,num_g=30,backend="jax"):
    GenericBackend.switchTo(backend)
    s = Inkstone()
    s.frequency = f
    s.lattice = 1
    s.num_g = num_g

    s.AddMaterial(name='di', epsilon=p)

    s.AddLayer(name='in', thickness=0, material_background='vacuum')
    s.AddLayer(name='slab', thickness=d, material_background='di')
    s.AddLayerCopy(name='out', original_layer='in', thickness=0)

    s.AddPattern1D(layer='slab', pattern_name='box', material='vacuum', width=w, center=0.5)

    s.SetExcitation(theta=0, phi=0, s_amplitude=1, p_amplitude=0)

    return s

def reflection_1layer_1d(d,w,f,p,num_g=30,backend="jax"):
    """
    Return m=0 reflection coefficient without diffraction
    """
    s = simulation_1layer_1d(d,w,f,p,num_g,backend)
    i, r = s.GetPowerFlux('in')
    return -r / i

def fields_1layer_1d(d,w,f,p,z,num_g=30,backend="jax"):
    """
    Returns fields at x=0, y=0, z=z
    """
    s = simulation_1layer_1d(d,w,f,p,num_g,backend)
    return s.GetFields(x=0,y=0,z=z)

def abs_Ey_1layer_1d(d,w,f,p,z,num_g=30,backend="jax"):
    """
    Returns abs(Ey) at (x,y,z)=(0,0,z)
    """
    GenericBackend.switchTo(backend)
    Ex, Ey, Ez, Hx, Hy, Hz = fields_1layer_1d(d,w,f,p,z,num_g,backend)
    return GenericBackend.genericBackend.abs(Ey[0][0][0])


## 2 layer 1d ##
def simulation_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,num_g=30,backend="jax"):
    GenericBackend.switchTo(backend)
    s = Inkstone()
    s.frequency = f
    s.lattice = 1
    s.num_g = num_g

    s.AddMaterial(name='di1', epsilon=p1)
    s.AddMaterial(name='di2', epsilon=p2)

    s.AddLayer(name='in', thickness=0, material_background='vacuum')
    s.AddLayer(name='slab1', thickness=d1, material_background='di1')
    s.AddLayer(name='slab2', thickness=d2, material_background='di2')
    s.AddLayerCopy(name='out', original_layer='in', thickness=0)

    s.AddPattern1D(layer='slab1', pattern_name='box1', material='vacuum', width=w1, center=0.5)
    s.AddPattern1D(layer='slab2', pattern_name='box2', material='vacuum', width=w2, center=0.5)

    s.SetExcitation(theta=theta, phi=0, s_amplitude=1, p_amplitude=0)

    return s

def reflection_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,order=-1,num_g=30,backend="jax"):
    """
    Return m=0 reflection coefficient with diffraction
    """
    GenericBackend.switchTo(backend)
    s = simulation_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,num_g,backend)
    inc_power = 0.5*1**2*GenericBackend.genericBackend.cos(theta*np.pi/180) # s_amp = 1 in simulation_2layer_1d
    order_power = s.GetPowerFluxByOrder(layer='in', order=order, z=-1) # z = -1 is the incidence region
    r_order_power = order_power[1] # t_order_power = order_power[0]
    r_order = GenericBackend.genericBackend.abs(r_order_power)/inc_power 
    return r_order

def fields_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,z,num_g=30,backend="jax"):
    """
    Returns fields at x=0, y=0, z=z
    """
    s = simulation_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,num_g,backend)
    return s.GetFields(x=0,y=0,z=z)

def abs_Ey_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,z,num_g=30,backend="jax"):
    """
    Returns abs(Ey) at (x,y,z)=(0,0,z)
    """
    GenericBackend.switchTo(backend)
    Ex, Ey, Ez, Hx, Hy, Hz = fields_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,z,num_g,backend)
    return GenericBackend.genericBackend.abs(Ey[0][0][0])


## 1 layer 2D ##
def simulation_1layer_2d(d,r,f,p,theta,phi,num_g=100,backend="jax"):
    GenericBackend.switchTo(backend)
    s = Inkstone()
    s.frequency = f
    s.lattice = ((1, 0), (0, 1))
    s.num_g = num_g

    s.AddMaterial(name='di', epsilon=p)

    s.AddLayer(name='in', thickness=0, material_background='vacuum')
    s.AddLayer(name='slab', thickness=d, material_background='di')
    s.AddLayerCopy(name='out', original_layer='in', thickness=0)

    s.AddPatternDisk(layer='slab', pattern_name='disk', material='vacuum', radius=r)

    s.SetExcitation(theta=theta, phi=phi, s_amplitude=1, p_amplitude=0)

    return s

def reflection_1layer_2d(d,r,f,p,theta,phi,num_g=100,backend="jax"):
    """
    Return m=0 reflection coefficient without diffraction
    """
    s = simulation_1layer_2d(d,r,f,p,theta,phi,num_g,backend)
    i, r = s.GetPowerFlux('in')
    return -r / i

def fields_1layer_2d(d,r,f,p,theta,phi,x,z,num_g=100,backend="jax"):
    """
    Returns fields at x=x, y=0, z=z
    """
    s = simulation_1layer_2d(d,r,f,p,theta,phi,num_g,backend)
    return s.GetFields(x=x,y=0,z=z)

def abs_Ey_1layer_2d(d,r,f,p,theta,phi,x,z,num_g=100,backend="jax"):
    """
    Returns abs(Ey) at (x,y,z)=(x,0,z)
    """
    GenericBackend.switchTo(backend)
    Ex, Ey, Ez, Hx, Hy, Hz = fields_1layer_2d(d,r,f,p,theta,phi,x,z,num_g,backend)
    return GenericBackend.genericBackend.abs(Ey[0][0][0])