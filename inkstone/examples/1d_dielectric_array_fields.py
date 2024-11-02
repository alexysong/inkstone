#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Dielectric array, with permittivity 12.
Period is 1
Each rod has side length 0.55.

 → x
↓z    ___     ___
...  |   |   |   |  ...
      ¯¯¯     ¯¯¯
"""
import torch

#确认numpy和torch的只算forward花费的时间
#当需要一个参数或多个参数gradient的时候，torch比numpy快多少
#多次计算求平均时间，减少误差
#credit paper
#from project_path import PATH as p
#import torch

#torch.set_printoptions(precision=20, threshold=10000)
import numpy as np

#np.set_printoptions(precision=5, threshold=100000)
#torch.autograd.set_detect_anomaly(True)
import sys
import time

sys.path.append('C:\\Users\\w-a-c\\Desktop\\inkstone\\')

start_time = time.time()

import inkstone.backends.BackendRegistry as bl

from inkstone.simulator import Inkstone
bk = bl.set_backend('numpy')


s = Inkstone()
s.lattice = 1.0

s.num_g = 20
s.frequency = 0.41

s.AddMaterial(name='di', epsilon=12.)

s.AddLayer(name='in', thickness=0., material_background='vacuum')

d = 0.55
s.AddLayer(name='slab', thickness=d, material_background='di')

s.AddLayerCopy(name='out', original_layer='in', thickness=0.)

s.AddPattern1D(layer='slab', pattern_name='box', material='vacuum', width=0.45, center=0.)

s.SetExcitation(theta=0., phi=0., s_amplitude=1., p_amplitude=0.)

Ex, Ey, Ez, Hx, Hy, Hz = s.GetFields(xmin=-0.5, xmax=0.5, nx=101,
                                    y=0.,
                                    zmin=-0.2, zmax=d + 0.2, nz=101)

#bk.get_gradient(Ey[0,0,1], )
print("%s" % (time.time() - start_time))
#print(s.lattice)
#print(s.lattice.grad[0,0])
#print(Ey[0, :, :].T)
#print(s.thicknesses['in'])
#print(s.thicknesses['out'].grad)
#for _, elem in enumerate(Ey.real.flatten()):
#    elem.backward(retain_graph=True)
#    print(s.lattice.grad)
'''
#%% plotting
from matplotlib import pyplot as plt

r = bk.abs(Ey[0, :, :]).detach()
#plt.plot(r)

plt.pcolormesh(bk.linspace(-0.5, 0.5, 101),
                bk.linspace(-0.2, d + 0.2, 101),
                r.T,
                shading='gouraud')
plt.xlabel('x')
plt.ylabel('z')
#plt.colorbar()

plt.show()
'''
