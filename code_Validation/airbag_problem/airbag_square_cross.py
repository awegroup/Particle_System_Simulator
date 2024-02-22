# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:10:12 2023

@author: Mark Kalsbeek

Verification example from: "An integrated aero-structural model for ram-air kite simulations" by Thedens. P.
URL: http://resolver.tudelft.nl/uuid:16e90401-62fc-4bc3-bf04-7a8c7bb0e2ee
Section 3.5, (Page 45)
"""
import sys, os
sys.path.append(os.path.abspath('../..'))

import numpy as np
import matplotlib.pyplot as plt

from src.particleSystem.ParticleSystem import ParticleSystem
from src.Sim.simulations import Simulate_airbag
import src.Mesh.mesh_functions as MF

params = {
    # model parameters
    "k": 250,  # [N/m]   spring stiffness
    "k_d": 250,  # [N/m] spring stiffness for diagonal elements
    "c": 10,  # [N s/m] damping coefficient
    "m_segment": 1, # [kg] mass of each node

    # Material parameters
    "E": 0.588, # [GPa] Stiffness of fabric
    "thickness": 0.6e-3, # [m]
    "rho": 333,# [kg/m3]

    # simulation settings
    "dt": 0.01,  # [s]       simulation timestep
    "t_steps": 1e4,  # [-]      number of simulated time steps
    "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
    "max_iter": 1e5,  # [-]       maximum number of iterations]
    "convergence_threshold": 1e-8, # [-]
    "min_iterations": 10, # [-]

    # sim specific settigns
    "pressure": 5e3 # [Pa] inflation pressure
    }

n_segments = 10
half_width = 422e-3 # [m]
diagonal_spring_ratio = 0.625 # Coefficient determined for this mesh shape to provide poissons ratio of 0.25
params['k'] = params["E"]*1e9*params['thickness']*n_segments / (n_segments+1+2*n_segments*diagonal_spring_ratio/np.sqrt(2))
params['k_d'] = params['k']*diagonal_spring_ratio

edge_length = half_width/n_segments

initial_conditions, connections = MF.mesh_airbag_square_cross(half_width,
                                                              edge_length ,
                                                              params = params,
                                                              noncompressive = True)
PS = ParticleSystem(connections, initial_conditions,params)

Sim = Simulate_airbag(PS, params)

if __name__ == '__main__':
    # Run coarse sim first with large timestep and masses
    Sim.run_simulation(plotframes=0,
                       plot_whole_bag = True,
                       printframes=10,
                       simulation_function = 'kinetic_damping')

    # Set correct masses and enable adaptive timestepping
    Sim.PS.params["adaptive_timestepping"] = edge_length*1e-2
    Sim.PS.params["convergence_threshold"] =1e-12
    Sim.PS.calculate_correct_masses(params['thickness'],params['rho'])
    Sim.run_simulation(plotframes=0,
                       plot_whole_bag = True,
                       printframes=10,
                       simulation_function = 'kinetic_damping')

    ax = PS.plot(colors='strain')
    ax.set_proj_type('ortho')