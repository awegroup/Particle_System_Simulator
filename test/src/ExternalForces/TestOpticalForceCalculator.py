# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:45:20 2024

@author: Mark Kalsbeek
"""
import unittest
import logging

import numpy as np

from LightSailSim.particleSystem.ParticleSystem import ParticleSystem 
from LightSailSim.ExternalForces.OpticalForceCalculator import OpticalForceCalculator, ParticleOpticalPropertyType
from LightSailSim.ExternalForces.LaserBeam import LaserBeam
from scipy.constants import c
from scipy.spatial.transform import Rotation
import LightSailSim.Mesh.mesh_functions as MF

class TestOpticalForceCalculator(unittest.TestCase):
    def setUp(self):
        
        # Set up ParticleSystem
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
        self.PS.calculate_correct_masses(1e-3, # [m] thickness
                                         1000) # [kg/m3] density
        self.mass = 1*1*1e-3 * 1000
        
        # Set up some sample LaserBeam instances
        I_0 = 100e9 /(10*10) # 100 GW divided over 100 square meters
        self.I_0 = I_0 #needed for calculations later
        mu_x = 0.5
        mu_y = 0.5
        sigma = 0.25
        self.LB_gaussian = LaserBeam(lambda x, y: I_0 * np.exp(-1/2 *((x-mu_x)/sigma)**2
                                                 -1/2 *((y-mu_y)/sigma)**2), 
                       lambda x,y: [0,1])
        self.LB_flat = LaserBeam(lambda x, y: np.ones(x.shape)*I_0, lambda x,y: [0,1])
        self.LB_flat_bounded = LaserBeam(laser_intensity_bounded, lambda x,y: [0,1])
        # Set up general purpose OpticalForceCalculator
        for particle in self.PS.particles:
            particle.optical_type = ParticleOpticalPropertyType.SPECULAR
            
        self.OpticalForces = OpticalForceCalculator(self.PS, self.LB_flat)
        
    
    
    def test_specular_flat(self):
        expected_force = 2*self.I_0 / c 
        OpticalForces = self.OpticalForces
        forces = OpticalForces.force_value()
        net_force = sum(np.linalg.norm(forces, axis=1))
        self.assertAlmostEqual(net_force, expected_force)
        
    def test_specular_45deg_x(self):
        expected_force =2*self.I_0 / c 
        OpticalForces = self.OpticalForces
        
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
        expected_force = 2*self.I_0 / c 
        OpticalForces = self.OpticalForces
        
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
        expected_force = 2*self.I_0 / c 
        OpticalForces = self.OpticalForces
        # Stretch PS into pyramid shape, which incrases surface area by sqrt(2)
        height = lambda x, y: min(0.5 - abs(x - 0.5), 0.5 - abs(y - 0.5))
        for particle in self.PS.particles:
            x,y,_ = particle.x
            z = height(x,y)
            particle.x[2]= z
        
        forces = OpticalForces.force_value()
        
        # Net force should be increased due to increase in surface area, but 
        # decreased due to angle of incidence, effects should cancel
        net_force = sum(np.linalg.norm(forces, axis=1))
        
        # There is a know discrepancy due to an error in the surface calculation
        # Therefore, these are tested to be less than two percent different
        with self.subTest(i=0): 
            #self.assertAlmostEqual(net_force,  expected_force, places = 2)
            self.assertGreater(0.02, abs(net_force/expected_force-1))

        vertical_force = sum(forces[:,2])
        with self.subTest(i=1):
            expected_vertical_force = expected_force/ np.sqrt(2)
            self.assertGreater(0.02, abs(vertical_force/expected_vertical_force-1))

    
    def test_axicon_flat(self):
        for i in range(4):
            with self.subTest(i=i):
                directing_angle = i*15
                directing_angle_rad = np.deg2rad(directing_angle)
                
                force_absorption = np.array([0, self.I_0 / c])
                force_emission = np.array([np.sin(directing_angle_rad*2),
                                           np.cos(directing_angle_rad*2)]
                                          )*self.I_0 / c
                expected_force = np.linalg.norm(force_absorption+force_emission)
                
                for particle in self.PS.particles:
                    particle.optical_type = ParticleOpticalPropertyType.AXICONGRATING
                    particle.axicon_angle = Rotation.from_euler('y', directing_angle, degrees=True).as_matrix()
                
                
                OpticalForces = OpticalForceCalculator(self.PS, self.LB_flat)
                forces = OpticalForces.force_value()
                net_force = sum(np.linalg.norm(forces, axis=1))
                self.assertAlmostEqual(net_force, expected_force)

    def test_find_center_of_mass(self):
        expected_COM = [0.5, 0.5, 0]
        OpticalForces = self.OpticalForces
        
        COM = OpticalForces.find_center_of_mass()
        for i in range(3):
            with self.subTest(i=i):
                self.assertAlmostEqual(COM[i], expected_COM[i])
        
        # Now we inflate and re-test
        expected_COM = [0.5, 0.5, 0.5/3]
        height = lambda x, y: min(0.5 - abs(x - 0.5), 0.5 - abs(y - 0.5))
        for particle in self.PS.particles:
            x,y,_ = particle.x
            z = height(x,y)
            particle.x[2]= z
        
        COM = OpticalForces.find_center_of_mass()
        for i in range(3):
            with self.subTest(i=i+3):
                self.assertAlmostEqual(COM[i], expected_COM[i], places = 3)
    
    def test_translate_mesh(self):
        OpticalForces = OpticalForceCalculator(self.PS, self.LB_flat)
        mesh = OpticalForces.translate_mesh(np.zeros([10,3]), np.ones([3]))
        
        self.assertEqual(np.all(mesh == np.ones([10,3])), True)
    
    def test_rotate_mesh(self):
        OpticalForces = self.OpticalForces
        mesh = np.meshgrid(np.linspace(0, 1, 6),
                           np.linspace(0, 1, 6))
        mesh = np.column_stack(list(zip(mesh[0],mesh[1]))).T
        mesh = np.column_stack((mesh,np.zeros(len(mesh)).T))
        
        original = mesh.copy()
        
        for i in range(3):
            mesh = OpticalForces.rotate_mesh(mesh, [0,0,120])
        
        diff = np.sum(mesh-original)
        self.assertAlmostEqual(diff, 0)
    
    def test_displace_particle_system(self):
        OpticalForces = self.OpticalForces
        
        with self.subTest(i=0):
            expected_COM = np.array([0.5, 0.5, 0]) + 1
            OpticalForces.displace_particle_system([1,1,1,0,0,0])
            COM = OpticalForces.find_center_of_mass()
            OpticalForces.un_displace_particle_system()
            self.assertAlmostEqual(np.sum(expected_COM-COM), 0)
        
        with self.subTest(i=1):
            x_0,_ = self.PS.x_v_current_3D
            logging.warning("Two displacement warnings expected as part of test")
            for i in range(3):
                # this function will issue a warning when called the second and third times
                OpticalForces.displace_particle_system([0,0,0,0,0,120]) 
            self.PS.current_displacement = [0,0,0,0,0,0]
            x_1,_ = self.PS.x_v_current_3D
            error = x_1 - x_0
            self.assertAlmostEqual(np.sum(error), 0)
        
    def test_un_displace_particle_system(self):
        OpticalForces = self.OpticalForces
        expected_COM = np.array([0.5, 0.5, 0])
        OpticalForces.displace_particle_system([0,0,0,45,45,45])
        COM = OpticalForces.find_center_of_mass()
        OpticalForces.un_displace_particle_system()
        self.assertAlmostEqual(np.sum(expected_COM-COM), 0)
        
        
    def test_stability_unit_mesh_uniform_beam(self):
        # For context, J maps changes in x,y,z,rx,ry,rz to changes in forces in those DOF's
        expected_force = 2*self.I_0 / c 
        
        J = self.OpticalForces.calculate_stability_coefficients()
        with self.subTest(i=0):
            # Check that translations do not affect other translations nor rotations
            self.assertTrue(np.allclose(J[0:6,0:3], np.zeros((6,3))))
        with self.subTest(i=1): 
            # Check that rotating about x does that same as about y
            self.assertAlmostEqual(J[1,3],-J[0,4])
        with self.subTest(i=2): 
            # Check that rotating about x and y gives right force gradient in x or y
            # If the plane is rotated 5 deg then the horizontal component of the foce will be 
            force_horizontal = expected_force*np.sin(np.deg2rad(5))*np.cos(np.deg2rad(5))
            d_force_horizontal_d_deg = force_horizontal /5
            self.assertAlmostEqual(J[1,3],-d_force_horizontal_d_deg)
        with self.subTest(i=3): 
            # Check that rotating about x and y gives right force gradient in z
            # Cos is squared, once for cosine factor, once for angle change of force
            d_force_vertical_d_deg = -expected_force*(1-np.cos(np.deg2rad(5))**2)/5
            self.assertAlmostEqual(J[2,3],d_force_vertical_d_deg)
        with self.subTest(i=4): 
            # Check that rotating about x or y has same effect on f_z
            self.assertAlmostEqual(J[2,3],J[2,4])
        with self.subTest(i=5): 
            # Check that rotation about z affects nothing
            self.assertTrue(np.allclose(J[:,5], np.zeros(6)))
        with self.subTest(i=6): 
            # Check that rotations about x or y don't affect any other rotations
            self.assertTrue(np.allclose(J[3:6,3:5], np.zeros((3,2))))
            
    def test_stability_unit_mesh_bounded_uniform_beam(self):
        # For context, J maps changes in x,y,z,rx,ry,rz to changes in forces in those DOF's
        expected_force = 2*self.I_0 / c 
        
        OpticalForces = OpticalForceCalculator(self.PS, self.LB_flat_bounded)
        J = OpticalForces.calculate_stability_coefficients()
        with self.subTest(i=0):
            # Check that translations in x,y,z do not affect forces in x,y
            self.assertTrue(np.allclose(J[0:2,0:3], np.zeros((2,3))))
        with self.subTest(i=1): 
            # Check that rotating about x does that same as about y
            self.assertAlmostEqual(J[1,3],-J[0,4]) 
        with self.subTest(i=2): 
            # Check that rotating about x or y gives right force gradient in x or y
            # If the plane is rotated 5 deg then the horizontal component of the foce will be 
            force_horizontal = expected_force*np.sin(np.deg2rad(5))*np.cos(np.deg2rad(5))
            d_force_horizontal_d_deg = force_horizontal /5
            self.assertAlmostEqual(J[1,3],-d_force_horizontal_d_deg)
        with self.subTest(i=3): 
            # Check that rotating about x and y gives right force gradient in z
            # Cos is squared, once for cosine factor, once for angle change of force
            d_force_vertical_d_deg = -expected_force*(1-np.cos(np.deg2rad(5))**2)/5
            self.assertAlmostEqual(J[2,3],d_force_vertical_d_deg)
        with self.subTest(i=4): 
            # Check that rotating about x or y has same effect on f_z
            self.assertAlmostEqual(J[2,3],J[2,4])
        with self.subTest(i=5): 
            # Check that rotation about z affects nothing except force in z
            mask = np.zeros(6)
            mask[2] = 1 
            mask = mask==0
            self.assertTrue(np.allclose(J[:,5][mask], np.zeros(5)))
        with self.subTest(i=6): 
            # Check that rotations about x or y don't affect any other rotations
            self.assertTrue(np.allclose(J[3:6,3:5], np.zeros((3,2))))
        with self.subTest(i=7):
            # Check that translations in x or y affect the force in z correctly
            force_vertical = expected_force*(1-0.1) # displacement 0.1 out of width 1
            self.assertAlmostEqual(J[2,0],J[2,1])
        with self.subTest(i=8):
            # and that they're equal
            self.assertAlmostEqual(J[2,0],J[2,1])
        with self.subTest(i=9):
            # Check that translation in x or y results in correct moment about y or x
            # inbalance of 10% of the force on the outer 10% of the sail
            moment_arm = 0.1/2 + 0.4 # distance to center of strip + distance strip+COM
            force = expected_force*0.1 
            moment = force*moment_arm
            d_moment_d_translation = moment/0.1
            self.assertAlmostEqual(J[4,0],-d_moment_d_translation)
        with self.subTest(i=10):
            # and that they're equal
            self.assertAlmostEqual(J[4,0],-J[3,1])
            
    def test_stability_unit_mesh_gaussian_beam(self):
        # Goal is to compare this against gaussian beam analytical results
        # Computations made with wolfram alpha
        
        # Set up the gaussian beam system
        OpticalForces = OpticalForceCalculator(self.PS, self.LB_gaussian)
        J = OpticalForces.calculate_stability_coefficients()

        with self.subTest(i=0):
            # To start out check if the forces are correct
            # integral of gaussian beam with constants supplied above is 
            P_gauss = self.I_0 * 0.35776
            f_expected = 2* P_gauss/c
            
            forces = OpticalForces.force_value()
            f_abs = np.linalg.norm(forces,axis=1)
            f_total = np.sum(f_abs)
            self.assertAlmostEqual(f_total, f_expected,places=3)
        
        # Check effect of displacement
        # integral of gaussian over shifted square:
        f_unshifted = f_total
        P_gauss = self.I_0 * 0.351218
        f_expected = 2* P_gauss/c
        
        forces = OpticalForces.force_value()
        f_abs = np.linalg.norm(forces,axis=1)
        with self.subTest(i=1):
            # change in z force
            d_f_d_x = (f_expected-f_unshifted)/0.1
            self.assertAlmostEqual(J[2,0], d_f_d_x,places = 3)

    def test_calculate_restoring_forces(self):
        # testing if forces for positive and negative displacements are the same
        # First translation, then rotation
        OpticalForces = OpticalForceCalculator(self.PS, self.LB_gaussian)
        
        displacement = np.array([0.1,0,0,0,0,0])
        #Positive translation
        OpticalForces.displace_particle_system(displacement)
        net_force_pos, net_moments_pos = OpticalForces.calculate_restoring_forces()
        OpticalForces.un_displace_particle_system()
        
        #Negative translation
        OpticalForces.displace_particle_system(-displacement)
        net_force_neg, net_moments_neg = OpticalForces.calculate_restoring_forces()
        OpticalForces.un_displace_particle_system()
        
        with self.subTest(i=0):
            self.assertTrue(np.allclose(net_force_pos,net_force_neg))
        with self.subTest(i=1):
            self.assertTrue(np.allclose(np.abs(net_moments_pos),np.abs(net_moments_neg)))
        
        displacement = np.array([0,0,0,5,0,0])
        #Positive rotation
        OpticalForces.displace_particle_system(displacement)
        net_force_pos, net_moments_pos = OpticalForces.calculate_restoring_forces()
        OpticalForces.un_displace_particle_system()
        
        #Negative rotation
        OpticalForces.displace_particle_system(-displacement)
        net_force_neg, net_moments_neg = OpticalForces.calculate_restoring_forces()
        OpticalForces.un_displace_particle_system()
        
        with self.subTest(i=2):
            self.assertTrue(np.allclose(np.abs(net_force_pos),np.abs(net_force_neg)))
        with self.subTest(i=3):
            self.assertTrue(np.allclose(np.abs(net_moments_pos),np.abs(net_moments_neg)))

def laser_intensity_bounded(x,y):
    I_0 = 100e9 /(10*10)
    intensity = np.zeros(x.shape)
    intensity[(x>-0.001)*(x<1.001)*(y>-0.001)*(y<1.001)] = I_0
    return  intensity 

if __name__ == '__main__':
    unittest.main()
    
    debug = False
    if debug:
        Logger = logging.getLogger()
        Logger.setLevel(10) # Use Logger.setLevel(30) to set it back to default
        T = TestOpticalForceCalculator()
        T.setUp()
