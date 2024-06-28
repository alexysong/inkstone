import numpy as np
import jax.numpy as jnp
import jax
from jax import grad as jax_grad
# jax.config.update("jax_debug_nans", True)

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

def abs_Ey_halfway(d,w,f,permittivity,num_g=30,backend="jax"):
    """
    Returns abs(Ey) at (x,y,z)=(0,0,d/2)
    """
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

    Ex, Ey, Ez, Hx, Hy, Hz = s.GetFields(x=0,y=0,z=d/2)
    
    return GenericBackend.genericBackend.abs(Ey[0][0][0])

eps = 1e-6
abs_tol = 1e-6
rel_tol = 1e-5

params_1d = (1.0, 0.4, 0.2, 12.)
d,w,f,permittivity = params_1d

# reflection_1d_grad0 = jax_grad(reflection_1d,0)
# reflection_1d_grad1 = jax_grad(reflection_1d,1)
# reflection_1d_grad2 = jax_grad(reflection_1d,2)
# reflection_1d_grad3 = jax_grad(reflection_1d,3)
# abs_Ey_halfway_grad0 = jax_grad(abs_Ey_halfway,0)
# abs_Ey_halfway_grad1 = jax_grad(abs_Ey_halfway,1)
# abs_Ey_halfway_grad2 = jax_grad(abs_Ey_halfway,2)
# abs_Ey_halfway_grad3 = jax_grad(abs_Ey_halfway,3)

# params = params_1d[1:]
# FD_grad = (reflection_1d(d+eps/2,*params,num_g=30,backend="jax") - reflection_1d(d-eps/2,*params,num_g=30,backend="jax"))/eps
# AD_grad = reflection_1d_grad0(d,*params,backend="jax")
# FD_grad = (abs_Ey_halfway(d+eps/2,*params,num_g=30,backend="jax") - abs_Ey_halfway(d-eps/2,*params,num_g=30,backend="jax"))/eps
# AD_grad = abs_Ey_halfway_grad0(d,*params,backend="jax")

# params = params_1d[2:]
# FD_grad = (reflection_1d(d,w+eps/2,*params,num_g=30,backend="jax") - reflection_1d(d,w-eps/2,*params,num_g=30,backend="jax"))/eps
# AD_grad = reflection_1d_grad1(d,w,*params,num_g=30,backend="jax")

# FD_grad = (reflection_1d(d,w,f+eps/2,permittivity,num_g=30,backend="jax") - reflection_1d(d,w,f-eps/2,permittivity,num_g=30,backend="jax"))/eps
# AD_grad = reflection_1d_grad2(d,w,f,permittivity,num_g=30,backend="jax")

# params = params_1d[:3]
# FD_grad = (reflection_1d(*params,permittivity+eps/2,num_g=30,backend="jax") - reflection_1d(*params,permittivity-eps/2,num_g=30,backend="jax"))/eps
# AD_grad = reflection_1d_grad3(*params,permittivity,num_g=30,backend="jax")

# assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

def simulation_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,num_g=30,backend="jax"):
    GenericBackend.switchTo(backend)
    s = NewInkstone()
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


# 2 layer 1D
def params_2layer_1d():
    d1 = 0.5
    d2 = 10.
    w1 = 0.4
    w2 = 0.9
    p1 = 12.
    p2 = 0.05
    f = 10.
    theta = 25. # degrees
    z = d1
    return [d1,d2,w1,w2,p1,p2,f,theta,z]

autodiffgrad = jax_grad(abs_Ey_2layer_1d, 8)

params_left = params_2layer_1d()[:8]
z = 0.5
# abs_Ey_2layer_1d(*params_left,z,num_g=30,backend="jax")
# print(abs_Ey_2layer_1d(*params_left,z,num_g=30,backend="jax"))

# d1 = params_left[0]
# d2 = params_left[1]
# z_positions = [-1., d1/2, d1, d1+d2/2, d1+d2, 1.05*(d1+d2)]
# for z in z_positions:
#     FD_grad = (abs_Ey_2layer_1d(*params_left,z+eps/2,num_g=30,backend="jax") 
#                 - abs_Ey_2layer_1d(*params_left,z-eps/2,num_g=30,backend="jax"))/eps
#     AD_grad = autodiffgrad(*params_left,z,num_g=30,backend="jax")
#     assert np.allclose(AD_grad, FD_grad, rtol=1000*rel_tol, atol=abs_tol)


autodiffgrad = jax_grad(reflection_2layer_1d, 7)
angles = jnp.linspace(-30.,30.,5, dtype=jnp.float64)
params_left = params_2layer_1d()[:7] 
theta = 30.
# print(reflection_2layer_1d(*params_left,theta))

# def refl(th,order):
#     return reflection_2layer_1d(*params_left,th,order=order)

# import matplotlib.pyplot as plt
# thetas = jnp.linspace(-30.,30.,100, dtype=jnp.float64)
# print(params_left)
# rNeg1 = jnp.array([refl(th,order=-1) for th in thetas])
# r1 = jnp.array([refl(th,order=1) for th in thetas])
# r0 = jnp.array([refl(th,order=0) for th in thetas])
# rTot = rNeg1+r1+r0
# plt.plot(thetas, rNeg1, ':r', label=r"$r_{-1}$")
# plt.plot(thetas, r1, ':b', label=r"$r_{1}$")
# plt.plot(thetas, r0, 'k', label=r"$r_{0}$")
# plt.plot(thetas, rTot, 'g', label=r"$r_{total}$")
# plt.legend()
# plt.show()

for th in angles:
    FD_grad = (reflection_2layer_1d(*params_left,th+eps/2,order=-1,num_g=30,backend="jax") 
                - reflection_2layer_1d(*params_left,th-eps/2,order=-1,num_g=30,backend="jax"))/eps
    AD_grad = autodiffgrad(*params_left,th,order=-1,num_g=30,backend="jax")
    assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

# nan -1.777129491620233