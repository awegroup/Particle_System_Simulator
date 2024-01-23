# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:01:32 2024

@author: Mark
"""
import unittest
import logging

import numpy as np

from src.particleSystem.ParticleSystem import ParticleSystem 
from src.Sim.simulations import SimulateTripleChainWithMass
from scipy.spatial.transform import Rotation
import src.Mesh.mesh_functions as MF

class TestParticleSystem(unittest.TestCase):
    def setUp(self):
        self.params = {
            # model parameters
            "k": 1,  # [N/m]   spring stiffness
            "k_d": 1,  # [N/m] spring stiffness for diagonal elements
            "c": 10,  # [N s/m] damping coefficient
            "m_segment": 1, # [kg] mass of each node
            
            # simulation settings
            "dt": 0.1,  # [s]       simulation timestep
            "t_steps": 1000,  # [-]      number of simulated time steps
            "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
            "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
            "max_iter": 1e4,  # [-]       maximum number of iterations
            "convergence_threshold": 1e-4, # [-]
            "min_iterations": 10, # [-]
            }
        self.initial_conditions, self.connectivity_matrix = MF.mesh_square(1, 1, 0.02, self.params)
        self.PS = ParticleSystem(self.connectivity_matrix, 
                                  self.initial_conditions, 
                                  self.params,
                                  clean_particles = False)
        
        self.PS.initialize_find_surface()
        
    def test_find_surface_flat(self):
        logging.debug('Debug notices for self.test_find_surface_flat')
        surfaces = self.PS.find_surface()
        total = np.sum(np.linalg.norm(surfaces,axis=1))
        self.assertAlmostEqual(total, 1.00)
        logging.debug('\n')
        
    def test_find_surface_pyramid(self):
        logging.debug('Debug notices for self.test_find_surface_pyramid')
        height = lambda x, y: min(0.5 - abs(x - 0.5), 0.5 - abs(y - 0.5))
        for particle in self.PS.particles:
            x,y,_ = particle.x
            z = height(x,y)
            particle.x[2]= z
        surfaces = self.PS.find_surface()
        total = np.sum(np.linalg.norm(surfaces,axis=1))
        self.assertAlmostEqual(total, 1.41, places = 2)
        logging.debug('\n')
        
    def test_constraint_point(self):
        logging.debug('Debug notices for self.test_constraint_point')
        initial_values = [
            [[-1, 0, 0],[0, 0, 0], 1, True],
            [[ 0, 0, 0],[0, 0, 0], 1, False],
            [[ 1, 0, 0],[0, 0, 0], 1, True]
            ]
        connectivity_matrix = [[0,1, 1, 1],
                                [1,2, 1, 1]
                                ]
        PS = ParticleSystem(connectivity_matrix, initial_values, self.params)

        PS.stress_self(0.8)
        for i in range(100):
            PS.simulate()
            
        logging.debug(str(PS))
        logging.debug('\n')
        
        x, y, z = PS.particles[0].x
        self.assertEqual(x, -1)
        x, y, z = PS.particles[2].x
        self.assertEqual(x, 1)
    
    def test_constraint_line(self):
        logging.debug('Debug notices for self.test_constraint_line')
        initial_values = [
            [[0, 0, 0],[0, 0, 0], 1, True],
            [[1, 0, 0],[0, 0, 0], 1, True, [1,-1,0], 'line'],
            [[0, 1, 0],[0, 0, 0], 1, True, [1,-1,0], 'line']
            ]
        connectivity_matrix = [[0,1, 1, 1],
                                [0,2, 1, 1],
                                [1,2, 1, 1]
                                ]
        PS = ParticleSystem(connectivity_matrix, initial_values, self.params)

        PS.stress_self(0.8)
        for i in range(100):
            PS.simulate()
            
        logging.debug(str(PS))
        logging.debug('\n')
        for particle in PS.particles[1:]:
            x, y, z = particle.x
            self.assertAlmostEqual(x+y, 1)
            
    def test_constraint_plane(self):
        logging.debug('Debug notices for self.test_constraint_plane')
        initial_values = [
            [[0, 0, 0],[0, 0, 0], 1, True],
            [[1, 0, 0],[0, 0, 0], 1, True, [1,1,0], 'plane'],
            [[0, 1, 0],[0, 0, 0], 1, True, [1,1,0], 'plane']
            ]
        connectivity_matrix = [[0,1, 1, 1],
                                [0,2, 1, 1],
                                [1,2, 1, 1]
                                ]
        PS = ParticleSystem(connectivity_matrix, initial_values, self.params)

        PS.stress_self(0.8)
        for i in range(100):
            PS.simulate()
            
        logging.debug(str(PS))
        logging.debug('\n')
        for particle in PS.particles[1:]:
            x, y, z = particle.x
            self.assertAlmostEqual(x+y, 1)

class TestParticleSystemReactionForces(unittest.TestCase):
    def setUp(self):
        self.params = {
            # model parameters
            "k": 1e5,  # [N/m]   spring stiffness
            "c": 1e3,  # [N s/m] damping coefficient
            "m_segment": 1, # [kg] mass of each node
            "proof_mass": 10, # [kg] mass of center node
            
            # simulation settings
            "dt": 0.001,  # [s]       simulation timestep
            "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
            "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
            "max_iter": 1e5,  # [-]       maximum number of iterations]
            "convergence_threshold": 1e-10, # Convergence value  treshold   
            "max_sim_steps": 1e4 # [-] Max timesteps
            }
        
        initial_conditions= []
        connectivity_matrix=[]
        # Create mesh: 
        # Connect three points that lay on an Isosceles trangle to the center
        # Let's do the vertical arm first and then just rotate it for the other ones
        n_segments_per_arm = 10
        n_points_per_arm = n_segments_per_arm+1
        self.params['m_particle'] = self.params['m_segment']/n_points_per_arm
        arm_length = 1
        y_coordinates = np.linspace(0,arm_length, n_points_per_arm)
        x_coordinates = np.zeros(y_coordinates.shape)
        rotation_matrix = Rotation.from_euler('z', 120, degrees=True).as_matrix()
        
        center_added = False
        for j in range(3): # Loop for each arm 
            # Add connections between center and arms
            connectivity_matrix.append([0,
                                           1+(n_points_per_arm)*j-j,
                                        self.params['k']*n_segments_per_arm,
                                        self.params['c']*n_segments_per_arm
                                        ])    
            for i in range(n_points_per_arm): # loop for each point
                if i != 0 and i!=1: # Add connections, but exclude some faults
                    connectivity_matrix.append([i+j*(n_points_per_arm-1),
                                                i+j*(n_points_per_arm-1)-1,
                                                self.params['k']*n_segments_per_arm,
                                                self.params['c']*n_segments_per_arm
                                                ])
                # Add points, fix outer ones and prevent duplicating the center
                if i == n_segments_per_arm: 
                    initial_conditions.append([[x_coordinates[i], y_coordinates[i], 0],
                                               np.zeros(3),
                                               self.params['m_particle']*1e5,
                                               True])
                else:
                    if not (center_added and x_coordinates[i]==0 and y_coordinates[i]==0):
                        initial_conditions.append([[x_coordinates[i], y_coordinates[i], 0],
                                                   np.zeros(3),
                                                   self.params['m_particle'],
                                                   False])
            center_added = True
            # Rotate arm coordinates
            new_coordinates = rotation_matrix.dot(np.vstack([x_coordinates, y_coordinates, np.zeros(x_coordinates.shape)])).T
            x_coordinates = new_coordinates[:,0]
            y_coordinates = new_coordinates[:,1]
            
           
    
        self.PS = ParticleSystem(connectivity_matrix, 
                                 initial_conditions, 
                                 self.params,
                                 clean_particles=False)
        self.PS.particles[0].set_m(self.params["proof_mass"])


    def test_find_reaction_forces(self):
        expected_vertical_force = (len(self.PS.particles)-4) * self.params["m_particle"]
        expected_vertical_force += self.params["proof_mass"]
        expected_vertical_force *= 9.81
        sim = SimulateTripleChainWithMass(self.PS,  self.params)
        sim.run_simulation()
        
        reactions = self.PS.find_reaction_forces()
        net_reactions = np.sum(reactions, axis=0) 
        """
        Okay so this don't work. The forces in the links connecting to the 
        particles that are fixed are for some reason way to high, in a way that
        all the other ones are not.
        I have a sneaking suspicion it might be due to the implementation of the 
        constraints in __system_jacobians so that'll be the first thing to try next
            because the length of those links are just slightly off I think its that
        The issue didn't manifest until I upped the stiffness, becasue that amplifies the error
        
        Debugging statements:
        """
        forces = [sd.force_value() for sd in self.PS.springdampers]
        force_magnitudes = np.linalg.norm(forces, axis=1)
        convergence_history = sim.convergence_history
        self.PS.plot()

        with self.subTest(i=0):
            self.assertAlmostEqual(net_reactions[0], net_reactions[1])
        with self.subTest(i=1):
            self.assertAlmostEqual(net_reactions[2], expected_vertical_force)

            
if __name__ == '__main__':
    unittest.main()
    
    debug = False
    if debug:
        import logging
        Logger = logging.getLogger()
        Logger.setLevel(00) # Use Logger.setLevel(30) to set it back to default
        T = TestParticleSystem()
        T.setUp()
        T.test_find_surface_flat()
        T.test_find_surface_pyramid()
        A=T.PS.find_surface()