#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
A photonic-crystal slab in vacuum. Permittivity 12.
Square lattice with period 1 in both x and y.
thickness 0.5
A circular hole of radius 0.2 in each unit cell.

 → y
↓x      ⋮
     ◯ ◯ ◯
...  ◯ ◯ ◯ ...  (top view)
     ◯ ◯ ◯
        ⋮
"""
import sys
sys.path.append("C:/Users/w-a-c/Desktop/inkstone")
import numpy as np
from inkstone import Inkstone

s = Inkstone()
s.lattice = ((1, 0), (0, 1))
s.num_g = 50

s.AddMaterial(name='di', epsilon=12)

s.AddLayer(name='in', thickness=0, material_background='vacuum')
s.AddLayer(name='slab', thickness=0.5, material_background='di')
s.AddLayerCopy(name='out', original_layer='in', thickness=0)

s.AddPatternDisk(layer='slab', pattern_name='disk', material='vacuum', radius=0.2)

# Incident wave
s.SetExcitation(theta=0, phi=0, s_amplitude=1, p_amplitude=0)

flux_in = []
flux_out = []
freq = np.linspace(0.25, 0.45, 201)
for i in freq:
    print('Frequency: {:g}'.format(i))

    s.SetFrequency(i)

    flux_in.append(s.GetPowerFlux('in'))
    flux_out.append(s.GetPowerFlux('out'))

incident = np.array([a[0] for a in flux_in])
reflection = -np.array([a[1] for a in flux_in]) / incident
transmission = np.array([a[0] for a in flux_out]) / incident

#%% plotting
from matplotlib import pyplot as plt
plt.plot(freq, transmission)
plt.xlabel('frequency')
plt.ylabel('transmission')
plt.show()

