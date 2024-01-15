# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:45:20 2024

@author: Mark Kalsbeek
"""
import unittest
import logging

import numpy as np

from src.particleSystem.ParticleSystem import ParticleSystem 
from src.ExternalForces.OpticalForceCalculator import OpticalForceCalculator, ParticleOpticalPropertyType
from src.ExternalForces.LaserBeam import LaserBeam
from scipy.constants import c
from scipy.spatial.transform import Rotation
import src.Mesh.mesh_functions as MF

class TestOpticalForceCalculator(unittest.TestCase):
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
        self.initial_conditions, self.connectivity_matrix = MF.mesh_square(1, 1, 0.05, self.params)
        self.PS = ParticleSystem(self.connectivity_matrix, 
                                 self.initial_conditions, 
                                 self.params,
                                 clean_particles = False)
        
        self.PS.initialize_find_surface() 
        
        I_0 = 100e9 /(10*10) # 100 GW divided over 100 square meters
        mu_x = 5
        mu_y = 5
        sigma = 5
        self.LB_gaussian = LaserBeam(lambda x, y: I_0 * np.exp(-1/2 *((x-mu_x)/sigma)**2
                                                 -1/2 *((y-mu_y)/sigma)**2), 
                       lambda x,y: [0,1])
        self.LB_flat = LaserBeam(lambda x, y: np.ones(x.shape)*I_0, lambda x,y: [0,1])
    
    def test_specular_flat(self):
        expected_force = 2*100e9/(10*10) / c 
        for particle in self.PS.particles:
            particle.optical_type = ParticleOpticalPropertyType.SPECULAR
            
        OpticalForces = OpticalForceCalculator(self.PS, self.LB_flat)
        forces = OpticalForces.force_value()
        net_force = sum(np.linalg.norm(forces, axis=1))
        self.assertAlmostEqual(net_force, expected_force)
        
    def test_specular_45deg_x(self):
        expected_force =2*100e9/(10*10) / c 
        
        # Assign optical properties
        for particle in self.PS.particles:
            particle.optical_type = ParticleOpticalPropertyType.SPECULAR
        
        # Init calculator 
        OpticalForces = OpticalForceCalculator(self.PS, self.LB_flat)
        
        # Rotate the PS, calculate forces, put it back 
        OpticalForces.displace_particle_system([0,0,0,45,0,0])
        forces = OpticalForces.force_value()
        net_force = sum(np.linalg.norm(forces, axis=1))
        OpticalForces.un_displace_particle_system()
        
        # Calculate summary numbers
        vertical_force = sum(forces[:,2])
        horizontal_force = sum(forces[:,1])
        null_force = sum(forces[:,0])
        
        # Perform checks
        # Scale by 1/sqrt(2) to account for cosine factor due to rotation
        with self.subTest(i=0):
            self.assertAlmostEqual(net_force,  expected_force/ np.sqrt(2))
        
        # Scale twice by 1/sqrt(2), once for cosine factor, once for vertical component
        with self.subTest(i=1): 
            self.assertAlmostEqual(vertical_force,expected_force/2)
        
        # Misc. checks
        with self.subTest(i=2): 
            self.assertAlmostEqual(abs(vertical_force), abs(horizontal_force))
        with self.subTest(i=3): 
            self.assertEqual(null_force,0)
    
    def test_specular_45deg_y(self):
        expected_force = 2*100e9/(10*10) / c 
        
        # Assign optical properties
        for particle in self.PS.particles:
            particle.optical_type = ParticleOpticalPropertyType.SPECULAR
        
        # Init calculator 
        OpticalForces = OpticalForceCalculator(self.PS, self.LB_flat)
        
        # Rotate the PS, calculate forces, put it back 
        OpticalForces.displace_particle_system([0,0,0,0,45,0])
        forces = OpticalForces.force_value()
        net_force = sum(np.linalg.norm(forces, axis=1))
        OpticalForces.un_displace_particle_system()
        
        # Calculate summary numbers
        vertical_force = sum(forces[:,2])
        horizontal_force = sum(forces[:,0])
        null_force = sum(forces[:,1])
        
        # Perform checks
        # Scale by 1/sqrt(2) to account for cosine factor due to rotation
        with self.subTest(i=0): 
            self.assertAlmostEqual(net_force,  expected_force/ np.sqrt(2))
        
        # Scale twice by 1/sqrt(2), once for cosine factor, once for vertical component
        with self.subTest(i=1): 
            self.assertAlmostEqual(vertical_force,expected_force/2)
        
        # Misc. checks
        with self.subTest(i=2): 
            self.assertAlmostEqual(abs(vertical_force), abs(horizontal_force))
        with self.subTest(i=3): 
            self.assertEqual(null_force,0)

    
    def test_specular_pyramid(self):
        expected_force = 2*100e9/(10*10) / c 

        # Stretch PS into pyramid shape, which incrases surface area by sqrt(2)
        height = lambda x, y: min(0.5 - abs(x - 0.5), 0.5 - abs(y - 0.5))
        for particle in self.PS.particles:
            x,y,_ = particle.x
            z = height(x,y)
            particle.x[2]= z
        
        for particle in self.PS.particles:
            particle.optical_type = ParticleOpticalPropertyType.SPECULAR
            
        OpticalForces = OpticalForceCalculator(self.PS, self.LB_flat)
        forces = OpticalForces.force_value()
        
        # Net force should be increased due to increase in surface area, but 
        # decreased due to angle of incidence, effects should cancel
        net_force = sum(np.linalg.norm(forces, axis=1))
        
        with self.subTest(i=0): 
            self.assertAlmostEqual(net_force,  expected_force, places = 2)

        vertical_force = sum(forces[:,2])
        with self.subTest(i=1):
            self.assertAlmostEqual(vertical_force,  expected_force/ np.sqrt(2))

    
    def test_axicon_flat(self):
        for i in range(4):
            with self.subTest(i=i):
                directing_angle = i*15
                directing_angle_rad = np.deg2rad(directing_angle)
                
                force_absorption = np.array([0, 100e9/(10*10) / c])
                force_emission = np.array([np.sin(directing_angle_rad*2),
                                           np.cos(directing_angle_rad*2)]
                                          )*100e9/(10*10) / c
                expected_force = np.linalg.norm(force_absorption+force_emission)
                
                for particle in self.PS.particles:
                    particle.optical_type = ParticleOpticalPropertyType.AXICONGRATING
                    particle.axicon_angle = Rotation.from_euler('y', directing_angle, degrees=True).as_matrix()
                
                
                OpticalForces = OpticalForceCalculator(self.PS, self.LB_flat)
                forces = OpticalForces.force_value()
                net_force = sum(np.linalg.norm(forces, axis=1))
                self.assertAlmostEqual(net_force, expected_force)

if __name__ == '__main__':
    unittest.main()
    
    debug = False
    if debug:
        Logger = logging.getLogger()
        Logger.setLevel(10) # Use Logger.setLevel(30) to set it back to default
        T = TestOpticalForceCalculator()
        T.setUp()
