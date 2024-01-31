# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:27:05 2024

@author: Mark Kalsbeek
"""
import numpy as np
import matplotlib.pyplot as plt
def ana_diag(x):
    return (1+0.401 * 2*x**2 + 1.1611 * x**4) * np.cos(np.pi*x/2)**2

#to be run after other sim converges
x, _ = PS.x_v_current_3D
n = int(np.sqrt(x.shape[0]))
x_diag = x[::n+1]
z_diag = x_diag[:,2]
z_diag_norm = z_diag/z_diag.max()
z_ana = ana_diag(np.linspace(-1,1,n))
dist = np.linspace(0,1,n)

plt.plot(dist,z_diag_norm, label='simulation result')
plt.plot(dist,z_ana,label='analytical result' )
plt.margins(0,0.1)
plt.legend()
plt.xlabel('Normalised distance on the diagonal')
plt.ylabel('Normalised displacement')