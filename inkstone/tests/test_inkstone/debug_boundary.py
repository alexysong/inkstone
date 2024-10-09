"""
Observing finite difference (wrt z position) divergence at boundary between two layers

"""

import inkstone.backends.BackendLoader as bl

bl.set_backend('jax')
bk = bl.backend()

from inkstone.simulator import Inkstone as NewInkstone # rename to NewInkstone so switchTo works correctly

import jax.numpy as jnp
from jax import grad as jax_grad

import matplotlib.pyplot as plt


# 1D 2 LAYER ################################################################################################################################################
### FUNCTIONS ###
def simulation_2layer_1d(d1,d2,w1,w2,p1,p2,f,theta,num_g=30):
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
    return [d1,d2,w1,w2,p1,p2,f,theta]



### PLOTTING FINITE DIFF VS STEP SIZE ###
num_g = 30
eps_vals = jnp.logspace(-10,-1,200)
d1 = params_2layer_1d()[0]
z_position = d1 # to measure |Ey|

def Ey_z(z):
    s = simulation_2layer_1d(*params_2layer_1d(),num_g)
    Ex, Ey, Ez, Hx, Hy, Hz = s.GetFields(x=0,y=0,z=z)
    return bk.abs(Ey[0][0][0])

def FD_Ey_z(z, eps):
    return (Ey_z(z+eps/2) - Ey_z(z-eps/2))/eps

# FDs = jnp.array([FD_Ey_z(z_position, eps) for eps in eps_vals])
# autodiffgrad = jax_grad(Ey_z)
# AD_grad = autodiffgrad(z_position)

# fig, ax = plt.subplots(1)
# ax.plot(eps_vals, FDs, '-k')
# ax.axhline(y=AD_grad, color='r', linestyle=':', label="Automatic derivative")
# ax.set_xscale("log")
# ax.set_yscale("symlog", linthresh=20)
# ax.set(xlabel=r"Finite difference step, $\epsilon$", ylabel="Finite difference at z = layer boundary")
# ax.legend()
# fig.tight_layout()
# plt.show()



params_left = params_2layer_1d()[:6]
params_right = params_2layer_1d()[7:]
f = 10.
z = d1/2
def Ey_f(f):
    s = simulation_2layer_1d(*params_left,f,*params_right,num_g)
    Ex, Ey, Ez, Hx, Hy, Hz = s.GetFields(x=0,y=0,z=z)
    return bk.abs(Ey[0][0][0])

def FD_Ey_f(f, eps):
    return (Ey_f(f+eps/2) - Ey_f(f-eps/2))/eps

FDs = jnp.array([FD_Ey_f(f, eps) for eps in eps_vals])
autodiffgrad = jax_grad(Ey_f)
AD_grad = autodiffgrad(f)

fig, ax = plt.subplots(1)
ax.plot(eps_vals, FDs, '-k')
ax.axhline(y=AD_grad, color='r', linestyle=':', label="Automatic derivative")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=20)
ax.set(xlabel=r"Finite difference step, $\epsilon$", ylabel=f"Finite difference for f = {f}")
ax.legend()
fig.tight_layout()
#fig.savefig("./Figures/FD_f_vs_eps.pdf")
plt.show()