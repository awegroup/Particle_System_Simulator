# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:10:12 2023

@author: Mark Kalsbeek
"""
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
    
    # simulation settings
    "dt": 0.01,  # [s]       simulation timestep
    "t_steps": 1000,  # [-]      number of simulated time steps
    "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
    "max_iter": 1e5,  # [-]       maximum number of iterations]
    "convergence_threshold": 1e-6, # [-]
    "min_iterations": 10, # [-]
    
    # sim specific settigns
    "pressure": 1e2 # [Pa] inflation pressure
    }

initial_conditions, connections = MF.mesh_airbag_square_cross(1, 1/10,params, noncompressive = True)
PS = ParticleSystem(connections, initial_conditions,params)

Sim = Simulate_airbag(PS, params)

if __name__ == '__main__':
    Sim.run_simulation(plotframes=0, 
                       plot_whole_bag = True,
                       printframes=50,
                       simulation_function = 'kinetic_damping')
    Sim.plot_whole_airbag()