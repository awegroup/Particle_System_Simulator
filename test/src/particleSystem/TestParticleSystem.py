# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:01:32 2024

@author: Mark
"""
import unittest

import numpy as np

from src.particleSystem.ParticleSystem import ParticleSystem 
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
            "max_iter": 1e5,  # [-]       maximum number of iterations]
            "convergence_threshold": 1e-6, # [-]
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
            
if __name__ == '__main__':
    unittest.main()
    
    debug = False
    if debug:
        import logging
        Logger = logging.getLogger()
        Logger.setLevel(10) # Use Logger.setLevel(30) to set it back to default
        T = TestParticleSystem()
        T.setUp()
        T.test_find_surface_flat()
        T.test_find_surface_pyramid()
        A=T.PS.find_surface()