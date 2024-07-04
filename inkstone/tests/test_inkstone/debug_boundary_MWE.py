from inkstone import Inkstone
import numpy as np
import matplotlib.pyplot as plt

def Ey_z(z):
    s = Inkstone()
    s.frequency = 10.
    s.lattice = 1
    s.num_g = 30

    s.AddMaterial(name='di1', epsilon=12.)
    s.AddMaterial(name='di2', epsilon=0.05)

    s.AddLayer(name='in', thickness=0, material_background='vacuum')
    s.AddLayer(name='slab1', thickness=0.5, material_background='di1')
    s.AddLayer(name='slab2', thickness=10., material_background='di2')
    s.AddLayerCopy(name='out', original_layer='in', thickness=0)

    s.AddPattern1D(layer='slab1', pattern_name='box1', material='vacuum', width=0.4, center=0.5)
    s.AddPattern1D(layer='slab2', pattern_name='box2', material='vacuum', width=0.9, center=0.5)

    s.SetExcitation(theta=25., phi=0, s_amplitude=1, p_amplitude=0)
    
    Ex, Ey, Ez, Hx, Hy, Hz = s.GetFields(x=0,y=0,z=z)
    return np.abs(Ey[0][0][0])

def FD_Ey_z(z, eps):
    return (Ey_z(z+eps/2) - Ey_z(z-eps/2))/eps
 
eps_vals = np.logspace(-10,-1,500)
d1 = 0.5 # layer 1 thickness
z_position = d1 # measure |Ey| at the layer boundary
FDs = np.array([FD_Ey_z(z_position, eps) for eps in eps_vals])

fig, ax = plt.subplots(1)
ax.plot(eps_vals, FDs, '-k')
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=20)
ax.set(xlabel=r"Finite difference step, $\epsilon$", ylabel="Finite difference at z = layer boundary")
fig.tight_layout()
plt.show()
