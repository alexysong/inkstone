import numpy as np
import matplotlib.pyplot as plt
from project_path import PATH as p
import torch

torch.set_printoptions(precision=20, threshold=10000)

import sys
import time

sys.path.append(p)
start_time = time.time()
import inkstone.backends.BackendLoader as bl

bl.set_backend('torch')
bk = bl.backend()
from inkstone.simulator import Inkstone

s = Inkstone()
s.lattice = 1.

s.num_g = 20
s.frequency = 0.41

s.AddMaterial(name='di', epsilon=12)

s.AddLayer(name='in', thickness=0., material_background='vacuum')

d = 0.55
s.AddLayer(name='slab', thickness=d, material_background='di')

s.AddLayerCopy(name='out', original_layer='in', thickness=0.)

s.AddPattern1D(layer='slab', pattern_name='box', material='vacuum', width=0.45, center=0.)

s.SetExcitation(theta=0., phi=0., s_amplitude=1., p_amplitude=0)

x = 1.
x_vals = np.linspace(x - 0.1, x + 0.1, 1000)
y_vals = []
for x_val in x_vals:
    s.lattice = x_val
    Ex, Ey, Ez, Hx, Hy, Hz = s.GetFields(xmin=-0.5, xmax=0.5, nx=101,
                                         y=0.,
                                         zmin=-0.2, zmax=d + 0.2, nz=101)# your function with x_temp
    y_vals.append(Ey[0,0,1].real.item())

plt.plot(x_vals, y_vals)
plt.axvline(x=1, color='r', linestyle='--')
plt.show()