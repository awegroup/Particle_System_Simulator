# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:21:49 2023

@author: Mark Kalsbeek

This module includes tools to aid in simulation, as well as some pre-baked simulation functions
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from src.particleSystem.ParticleSystem import ParticleSystem 
import src.Mesh.mesh_functions as MF



class Simulate:
    def __init__(self, ParticleSystem):
        self.PS = ParticleSystem
    
    
class Simulate_1d_Stretch(Simulate):
    def __init__(self, ParticleSystem, sim_params, save_plots = False):
        self.PS = ParticleSystem
        self.params = sim_params
        self.steps = sim_params['steps'] # represent strain values 
        self.history = {}
        
    def run_simulation(self):
        x_cleaned = np.array([particle.x for particle in self.PS.particles])
        starting_dimentions = np.ptp(x_cleaned, axis = 0)
        
        midstrip_indices = MF.ps_find_mid_strip_y(self.PS, 
                                                  self.params['midstrip_width'])
        
        self.PS, boundaries = MF.ps_fix_opposite_boundaries_x(self.PS, 
                                                              margin = self.params['boundary_margin'])
        
        
        total_displacement = 0
        for strain in self.steps:
            starting_time = time.time()
            print(f'Starting simulation on step {strain=}')
            displacement = starting_dimentions[0] * strain - total_displacement
            total_displacement += displacement
            MF.ps_stretch_in_x(self.PS, boundaries[1], displacement)
            
            
            converged = False
            convergence_history = []
            while not converged:
                PS.simulate()
                    
                #convergence check
                ptp_range = MF.ps_find_strip_dimentions(self.PS, midstrip_indices)
                transverse_strain = (ptp_range[1]-starting_dimentions[1])/starting_dimentions[1]
                
                reaction_force = MF.ps_find_reaction_of_boundary(self.PS, boundaries[1])
                reaction_force = np.linalg.norm(reaction_force)
                
                e_kin = self.PS.kinetic_energy
                convergence_history.append(e_kin)
                
                if len(convergence_history)>5:
                    if abs(convergence_history[-1]-convergence_history[-2]) < 1e-15:
                        converged = True
                        
            finished_time = time.time()
            delta_time = finished_time - starting_time
            
            
            poissons_ratio = transverse_strain / strain
            
            self.history[strain] = [reaction_force, poissons_ratio]
            
            print(f'Finished with {transverse_strain=:.2f} and force {reaction_force=:.2f}')
            print(f'That took  {delta_time//60:.0f}m {delta_time%60:.2f}s')
            print('\n')

            
    def plot_results(self):
        reaction_force = []
        poissons_ratio = []
        
        for step in self.steps:
            force, ratio = self.history[step]
            reaction_force.append(force)
            poissons_ratio.append(ratio)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(self.steps, reaction_force)
        ax1.set_title("Reaction Force versus Strain")
        
        ax2 = fig.add_subplot(122)
        ax2.plot(self.steps, poissons_ratio)
        ax2.set_title("Poissons Ratio versus Strain")
                
                
                
if __name__ == '__main__':
    params = {
        # model parameters
        "k": 1,  # [N/m]   spring stiffness
        "k_d": 1,  # [N/m] spring stiffness for diagonal elements
        "c": 1,  # [N s/m] damping coefficient
        "m_segment": 1, # [kg] mass of each node
        
        # simulation settings
        "dt": 0.1,  # [s]       simulation timestep
        "t_steps": 1000,  # [-]      number of simulated time steps
        "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
        "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
        "max_iter": 1e5,  # [-]       maximum number of iterations]
        
        # Simulation Steps
        "steps": np.linspace(0.01,0.5, 15),
        
        # Mesh_dependent_settings
        "midstrip_width": 1,
        "boundary_margin": 0.175
        }
    
    initial_conditions, connections = MF.mesh_square_cross(20,20,1,params)
    initial_conditions, connections = MF.mesh_rotate_and_trim(initial_conditions, 
                                                           connections, 
                                                           45/2)    
    PS = ParticleSystem(connections, initial_conditions,params)

    Sim = Simulate_1d_Stretch(PS, params)
    Sim.run_simulation()
    Sim.plot_results()
    

