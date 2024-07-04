#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Dielectric array, with permittivity 12.
Period is 1
Each rod has side length 0.55.

 → x
↓z    ___     ___
...  |   |   |   |  ...  (side view)
      ¯¯¯     ¯¯¯
"""
import sys
from project_path import PATH as p
sys.path.append(p)
from inkstone import Inkstone
from inkstone.backends.GenericBackend import genericBackend as gb
import inkstone.backends.GenericBackend as gb2
#gb2.switchTo('torch')
s = Inkstone()
s.lattice = 1
s.num_g = 30

s.AddMaterial(name='di', epsilon=12)

s.AddLayer(name='in', thickness=0, material_background='vacuum')
d = 0.55
s.AddLayer(name='slab', thickness=d, material_background='di')
s.AddLayerCopy(name='out', original_layer='in', thickness=0)

s.AddPattern1D(layer='slab', pattern_name='box', material='vacuum', width=0.45, center=0.5)

s.SetExcitation(theta=0, phi=0, s_amplitude=1, p_amplitude=0)

I = []
R = []
T = []
#freq = torch.linspace(0.2, 0.9, 201)
freq = gb.linspace(0.2, 0.9, 201)
for f in freq:
    s.frequency = f
    i, r = s.GetPowerFlux('in')
    I.append(i)
    R.append(-r / i)
    T.append(s.GetPowerFlux('out')[0] / i)
    print("frequency: {:g}".format(f))

#%% plotting
from matplotlib import pyplot as plt

plt.figure()
plt.plot(freq, R)
plt.xlabel('frequency')
plt.ylabel('reflection')
plt.figure()
plt.plot(freq, T)
plt.xlabel('frequency')
plt.ylabel('transmission')
plt.show()