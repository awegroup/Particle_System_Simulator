# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:21:49 2023

@author: Mark Kalsbeek

This module includes tools to aid in simulation, as well as some pre-baked simulation functions
"""

import time
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from src.particleSystem.ParticleSystem import ParticleSystem 
import src.Mesh.mesh_functions as MF



class Simulate:
    def __init__(self, ParticleSystem):
        self.PS = ParticleSystem
    
    def run_simulation(self):
        pass
    
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
                


class Simulate_airbag(Simulate):
    def __init__(self, ParticleSystem, params):
        self.PS = ParticleSystem
        self.params = params
        self.pressure = params['pressure']            # [Pa]

        
    def run_simulation(self, 
                       plotframes: int = 0,
                       plot_whole_bag: bool = False,
                       printframes: int = 10,
                       simulation_function: str = 'default'
                       ):
        """
        

        Parameters
        ----------
        plotframes : INT, optional
            Save every nth frame. The default is 0, indicating saving zero frame.
        plot_whole_bag : bool, optional
            Wether or not to plot the whole bag. The default is False.
        printframes : int, optional
            Print a mesage every nth frame. The default is 10.
        simulation_function : str, optional
            Allows enabling kinetic damping by passing 'kinetic_damping'. The default is 'default'.

        Returns
        -------
        None.

        """
        if simulation_function == 'kinetic_damping':
            simulation_function = self.PS.kin_damp_sim
        else:
            simulation_function = self.PS.simulate
        
        
        converged = False
        convergence_history = []
        dt = self.params['dt']
        
        if plotframes:
            fig = plt.figure()
        step = 0
        start_time = time.time()
        last_time = time.time()

        while not converged:
            step+= 1
            if plotframes and step%plotframes==0:
                ax = fig.add_subplot(projection='3d')
                
                if plot_whole_bag:
                    self.plot_whole_airbag(ax)
                else: 
                    self.PS.plot(ax)
                ax.set_title(f"Simulate_airbag, t = {step*dt:.3f}")
                fig.tight_layout()
                fig.savefig(f'temp\Airbag{step}.jpg', dpi = 200, format = 'jpg')
                fig.clear()
            
            
            areas = self.PS.find_surface()
            areas = np.nan_to_num(areas)
            f = np.hstack(areas) * self.pressure
            
            simulation_function(f)
            
            d_crit_d_step = 0
            convergence_history.append(self.PS.kinetic_energy)
            if  len(convergence_history)>self.params['min_iterations']:
                d_crit_d_step = abs(convergence_history[-1]-convergence_history[-2])
                if d_crit_d_step<self.params['convergence_threshold']:
                    converged = True
                
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            if printframes and step%printframes==0:
                print(f'Just finished step {step}, it took {delta_time//60:.0f}m {delta_time%60:.2f}s, {d_crit_d_step=:.2g}')
        delta_time = current_time - start_time
        print(f'Converged in {delta_time//60:.0f}m {delta_time%60:.2f}s')
        fig_converge = plt.figure()
        ax1 = fig_converge.add_subplot()
        ax1.semilogy(convergence_history)
        ax1.set_title('Convergence History')
        print(convergence_history)
        
    def plot_whole_airbag(self, 
                          ax = None,
                          plotting_function = 'default'):
        """
        Plotting function that rotates and mirrors the simulated section 

        Parameters
        ----------
        ax : matplotlib axis object
            Checks for preexisting axis. If none is given, one is made
        plotting_function : TYPE, optional
            Allows for surface plot of the bag by passing 'surface'. The default is 'default'.

        Returns
        -------
        ax : TYPE
            DESCRIPTION.

        """
        plotfunct_dict = {'default': self.PS.plot,
                          'surface': self.PS.plot_triangulated_surface}
        
        plotting_function = plotfunct_dict[plotting_function]
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            
        PS = self.PS
        
        x, _ = PS.x_v_current

        rotation_matrix = sp.spatial.transform.Rotation.from_euler('z', 90, degrees=True).as_matrix()
        rotation_matrix = sp.linalg.block_diag(*[rotation_matrix for i in range(int(len(x)/3))])
        
        for i in range(4):
            
            x = rotation_matrix.dot(x)
            PS.update_pos_unsafe(x)
            plotting_function(ax)
            x[2::3] *= -1
            PS.update_pos_unsafe(x)
            plotting_function(ax)
        
        return ax

            
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
    

