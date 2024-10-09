"""
Debugging inkstone_objective_functions.py functions and derivatives

"""

import numpy as np
import jax.numpy as jnp
import jax
from jax import grad as jax_grad
# jax.config.update("jax_debug_nans", True)

import inkstone.backends.BackendLoader as bl

bl.set_backend('jax')
bk = bl.backend()
from inkstone.simulator import Inkstone as NewInkstone # rename to NewInkstone so switchTo works correctly

import matplotlib.pyplot as plt



# GLOBAL SETTINGS ################################################################################################################################################
eps = 1e-6
abs_tol = 1e-6
rel_tol = 1e-5



# 1D 1 LAYER ################################################################################################################################################
### FUNCTIONS ###
def simulation_1layer_1d(d,w,f,p,num_g=30,backend="jax"):

    s = NewInkstone()
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
    Ex, Ey, Ez, Hx, Hy, Hz = fields_1layer_1d(d,w,f,p,z,num_g,backend)
    # return jnp.real(Ey[0][0][0])
    return bk.abs(Ey[0][0][0])



### PARAMETERS ###
def params_1layer_1d():
    d = 1.
    w = 0.4
    f = 0.2
    p = 12.
    z = 1/2
    return [d,w,f,p,z]



### TEST FD VS AD ###
# Verified real(Ey) vs z: expect sinusoidal Ey inside grating, perfect Ey sines outside the grating
# Verified |Ey| vs z: expect sinusoidal |Ey| inside grating, constant |Ey| in output region and sinusoidal |Ey| in incidence region
# Sinusoidal Ey inside grating comes from the Fourier (sine) waves mixing together
# |Ey| is sinusoidal in incidence region because the incident wave mixes with the reflected wave
# params_left = params_1layer_1d()[:4]
# def Ey(z):
#     return abs_Ey_1layer_1d(*params_left,z,num_g=30,backend="jax")
# z_positions = jnp.linspace(-5.,5.,200, dtype=jnp.float64) 
# Eys = jnp.array([Ey(z) for z in z_positions])
# plt.plot(z_positions, Eys, '-k')
# plt.show()

# |Ey| vs d: for z fixed (or proportional to d), |Ey| is periodic in d
# Likely comes from the exp(iq(d-z)) dependence
# params_right = params_1layer_1d()[1:]
# def Ey(d):
#     return abs_Ey_1layer_1d(d,*params_right,num_g=30,backend="jax")
# grating_thicknesses = jnp.linspace(1,10.,200, dtype=jnp.float64) 
# Eys = jnp.array([Ey(d) for d in grating_thicknesses])
# plt.plot(grating_thicknesses, Eys, '-k')
# plt.show()



# 1D 2 LAYER ################################################################################################################################################
### FUNCTIONS ###
def simulation_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,num_g=30,backend="jax"):
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
    s = simulation_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,num_g,backend)
    inc_power = 0.5*1**2*bl.cos(theta*np.pi/180) # s_amp = 1 in simulation_2layer_1d
    order_power = s.GetPowerFluxByOrder(layer='in', order=order, z=-1) # z = -1 is the incidence region
    r_order_power = order_power[1] # t_order_power = order_power[0]
    r_order = bk.abs(r_order_power)/inc_power
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
    Ex, Ey, Ez, Hx, Hy, Hz = fields_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,z,num_g,backend)
    return bk.abs(Ey[0][0][0])



### PARAMETERS ###
def params_2layer_1d():
    d1 = 0.5
    d2 = 10.
    w1 = 0.4
    w2 = 0.9
    p1 = 12.
    p2 = 0.05
    f = 10.
    theta = 25. # degrees
    z = 1.05*d1
    return [d1,d2,w1,w2,p1,p2,f,theta,z]



### TESTING FD VS AD ###
## Ey vs z position ##
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


## Reflection vs theta ##
autodiffgrad = jax_grad(reflection_2layer_1d, 7)
angles = jnp.linspace(-30.,30.,5, dtype=jnp.float64)
params_left = params_2layer_1d()[:7] 
theta = 30.
# print(reflection_2layer_1d(*params_left,theta))

# def refl(th,order):
#     return reflection_2layer_1d(*params_left,th,order=order)

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

# for th in angles:
#     FD_grad = (reflection_2layer_1d(*params_left,th+eps/2,order=-1,num_g=30,backend="jax") 
#                 - reflection_2layer_1d(*params_left,th-eps/2,order=-1,num_g=30,backend="jax"))/eps
#     AD_grad = autodiffgrad(*params_left,th,order=-1,num_g=30,backend="jax")
#     assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)

# nan -1.777129491620233


## Ey vs layer 2 thickness ##
# autodiffgrad = jax_grad(abs_Ey_2layer_1d, 1)
# grating_thicknesses = jnp.linspace(0.1,100,5, dtype=jnp.float64) # FD diverges at d2 = 0.1
grating_thicknesses = jnp.linspace(1,100,5, dtype=jnp.float64) 
params_left = params_2layer_1d()[:1] 
params_right = params_2layer_1d()[2:]
eps = 1e-7
num_g = 30

def Ey(d):
    return abs_Ey_2layer_1d(*params_left,d,*params_right,num_g=num_g,backend="jax")

grating_thicknesses = jnp.linspace(1,10,200, dtype=jnp.float64) 
Eys = jnp.array([Ey(d) for d in grating_thicknesses])
plt.plot(grating_thicknesses, Eys, '-k')
plt.show()

# for d2 in grating_thicknesses:
#     Ey_plus = abs_Ey_2layer_1d(*params_left,d2+eps/2,*params_right,num_g=num_g,backend="jax")
#     Ey_minus = abs_Ey_2layer_1d(*params_left,d2-eps/2,*params_right,num_g=num_g,backend="jax")
#     FD_grad = (Ey_plus - Ey_minus)/eps
#     AD_grad = autodiffgrad(*params_left,d2,*params_right,num_g=num_g,backend="jax")
#     print(f"Ey(d2+eps/2) = {Ey_plus}\nEy(d2-eps/2) = {Ey_minus}\nFD_grad = {FD_grad}\nAD_grad = {AD_grad}")
#     assert np.allclose(AD_grad, FD_grad, rtol=rel_tol, atol=abs_tol)