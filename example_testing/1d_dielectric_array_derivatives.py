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
sys.path.append("../")
sys.path.append("../inkstone")
import GenericBackend
GenericBackend.switchTo("jax")

from inkstone import Inkstone
import numpy as np
from jax import grad

def calc_reflection(d,w,f,permittivity):
      s = Inkstone()
      s.frequency = f
      s.lattice = 1
      s.num_g = 30

      s.AddMaterial(name='di', epsilon=permittivity)

      s.AddLayer(name='in', thickness=0, material_background='vacuum')
      s.AddLayer(name='slab', thickness=d, material_background='di')
      s.AddLayerCopy(name='out', original_layer='in', thickness=0)

      s.AddPattern1D(layer='slab', pattern_name='box', material='vacuum', width=w, center=0.5)

      s.SetExcitation(theta=0, phi=0, s_amplitude=1, p_amplitude=0)

      i, r = s.GetPowerFlux('in')
      return -r / i

d = 0.55
w = 0.45
permittivity = 12
R = []
freq = np.linspace(0.2, 0.9, 201)
grad_reflection_wrt_d = grad(calc_reflection,0)
print(grad_reflection_wrt_d(d,w,freq[0],permittivity))
# for f in freq:
#     R.append(calc_reflection(d,w,f,permittivity))

# # plotting
# from matplotlib import pyplot as plt

# plt.figure()
# plt.plot(freq, R)
# plt.xlabel('frequency')
# plt.ylabel('reflection')
# plt.show()