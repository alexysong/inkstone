import numpy as np
import matplotlib.pyplot as plt
#from project_path import PATH as p
import torch

torch.set_printoptions(precision=20, threshold=10000)

import sys
import time

#sys.path.append(p)
sys.path.append('C:\\Users\\w-a-c\\Desktop\\inkstone\\')
import inkstone.backends.BackendRegistry as bl

x = 2.
x_vals = np.linspace(x-1, x+1, 30)

#x_vals = [0.5,0.6,0.7,0.8,0.9]
def getEy(bk, x_vals):

    bl.set_backend(bk)

    from inkstone.simulator import Inkstone as Inkstone
    y_avals = []
    ti = []
    for x_val in x_vals:
        start_time = time.time()
        s = Inkstone()
        s.lattice = x_val

        s.num_g = 20
        s.frequency = 0.41

        s.AddMaterial(name='di', epsilon=12.)

        s.AddLayer(name='in', thickness=0., material_background='vacuum')

        d = 0.55
        s.AddLayer(name='slab', thickness=d, material_background='di')

        s.AddLayerCopy(name='out', original_layer='in', thickness=0.)

        s.AddPattern1D(layer='slab', pattern_name='box', material='vacuum', width=0.45, center=0.)

        s.SetExcitation(theta=0., phi=0., s_amplitude=1., p_amplitude=0)

        Ex, Ey, Ez, Hx, Hy, Hz = s.GetFields(xmin=-0.5, xmax=0.5, nx=101,
                                             y=0.,
                                             zmin=-0.2, zmax=d + 0.2, nz=101)  # your function with x_temp
        ti.append(time.time() - start_time)
        """
        if bk == 'torch':
            Ey[0, 0, 1].real.backward()
            y_avals.append(s.lattice.grad[0,0])
            #s.materials['di'].epsi.grad.zero_()
        else:
            y_avals.append(Ey[0, 0, 1])"""
    print(np.average(ti))
    return y_avals


y_avals = getEy('torch', x_vals)
y_vals = getEy('numpy', x_vals)
'''
#y = np.gradient(y_vals, x_vals)
#print(x_vals)
#print([x.item() for x in y_avals])
#print([x.real for x in y_vals])
plt.plot(x_vals, y, color='blue', label="np.gradient")
plt.plot(x_vals, y_avals, color='orange', label="torch.autograd")
plt.axvline(x=1, color='r', linestyle='--')
plt.title('np.gradient vs torch.autograd')
plt.xlabel('x')
plt.ylabel('Gradient')
plt.legend()
plt.show()
'''