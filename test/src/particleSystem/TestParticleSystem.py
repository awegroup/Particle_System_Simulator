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
        params = {
            # model parameters
            "k": 1,  # [N/m]   spring stiffness
            "k_d": 1,  # [N/m] spring stiffness for diagonal elements
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
            }
        self.initial_conditions, self.connectivity_matrix = MF.mesh_square(1, 1, 0.02, params)
        self.PS = ParticleSystem(self.connectivity_matrix, 
                                 self.initial_conditions, 
                                 params,
                                 clean_particles = False)
        
        self.PS.initialize_find_surface()
        
    def test_find_surface_flat(self):
        surfaces = self.PS.find_surface()
        total = np.sum(np.linalg.norm(surfaces,axis=1))
        self.assertAlmostEqual(total, 1.00)
        
    def test_find_surface_pyramid(self):
        height = lambda x, y: min(0.5 - abs(x - 0.5), 0.5 - abs(y - 0.5))
        for particle in self.PS.particles:
            x,y,_ = particle.x
            z = height(x,y)
            particle.x[2]= z
        surfaces = self.PS.find_surface()
        total = np.sum(np.linalg.norm(surfaces,axis=1))
        self.assertAlmostEqual(total, 1.41, places = 2)
        

            
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