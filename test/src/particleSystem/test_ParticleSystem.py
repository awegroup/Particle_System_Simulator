import numpy as np
from scipy.spatial.transform import Rotation
import pytest

from LightSailSim.particleSystem.ParticleSystem import ParticleSystem 
from LightSailSim.Sim.simulations import SimulateTripleChainWithMass
import LightSailSim.Mesh.mesh_functions as MF

@pytest.fixture
def params():
      return {
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

@pytest.fixture
def PS(params):
    mesh = MF.mesh_square(1, 1, 0.02, params)
    PS = ParticleSystem(*mesh, params, clean_particles = False)
    PS.initialize_find_surface()
    return PS

def test_find_surface_flat(PS):
        # logging.debug('Debug notices for self.test_find_surface_flat')
        surfaces = PS.find_surface()
        total = np.sum(np.linalg.norm(surfaces,axis=1))
        np.testing.assert_allclose(total, 1.00)
        # logging.debug('\n')

def test_find_surface_pyramid(PS):
        # There is a known error due to the triangulation, therefore this test
        # is comparing percent different
        # logging.debug('Debug notices for self.test_find_surface_pyramid')
        height = lambda x, y: min(0.5 - abs(x - 0.5), 0.5 - abs(y - 0.5))
        for particle in PS.particles:
            x,y,_ = particle.x
            z = height(x,y)
            particle.x[2]= z
        surfaces = PS.find_surface()
        total = np.sum(np.linalg.norm(surfaces,axis=1))
        # logging.debug('\n')
        assert 0.01 > total/1.41-1

def test_constraint_point(params):
        # logging.debug('Debug notices for self.test_constraint_point')
        initial_values = [
            [[-1, 0, 0],[0, 0, 0], 1, True],
            [[ 0, 0, 0],[0, 0, 0], 1, False],
            [[ 1, 0, 0],[0, 0, 0], 1, True]
            ]
        connectivity_matrix = [[0,1, 1, 1],
                                [1,2, 1, 1]
                                ]
        PS = ParticleSystem(connectivity_matrix, initial_values, params, init_surface=False)

        PS.stress_self(0.8)
        for _ in range(100):
            PS.simulate()
            
        # logging.debug(str(PS))
        # logging.debug('\n')
        
        x, y, z = PS.particles[0].x
        assert x == -1
        x, y, z = PS.particles[2].x
        assert x == 1

def test_constraint_line(params):
    # logging.debug('Debug notices for self.test_constraint_line')
    initial_values = [
        [[0, 0, 0],[0, 0, 0], 1, True],
        [[1, 0, 0],[0, 0, 0], 1, True, [1,-1,0], 'line'],
        [[0, 1, 0],[0, 0, 0], 1, True, [1,-1,0], 'line']
        ]
    connectivity_matrix = [[0,1, 1, 1],
                            [0,2, 1, 1],
                            [1,2, 1, 1]
                            ]
    PS = ParticleSystem(connectivity_matrix, initial_values, params, init_surface=False)

    PS.stress_self(0.8)
    for _ in range(100):
        PS.simulate()
        
    # logging.debug(str(PS))
    # logging.debug('\n')
    for particle in PS.particles[1:]:
        x, y, z = particle.x
        np.testing.assert_allclose(x+y, 1)

def test_constraint_plane(params):
    # logging.debug('Debug notices for self.test_constraint_plane')
    initial_values = [
        [[0, 0, 0],[0, 0, 0], 1, True],
        [[1, 0, 0],[0, 0, 0], 1, True, [1,1,0], 'plane'],
        [[0, 1, 0],[0, 0, 0], 1, True, [1,1,0], 'plane']
        ]
    connectivity_matrix = [[0,1, 1, 1],
                            [0,2, 1, 1],
                            [1,2, 1, 1]
                            ]
    PS = ParticleSystem(connectivity_matrix, initial_values, params, init_surface=False)

    PS.stress_self(0.8)
    for i in range(100):
        PS.simulate()
        
    # logging.debug(str(PS))
    # logging.debug('\n')
    for particle in PS.particles[1:]:
        x, y, z = particle.x
        np.testing.assert_allclose(x+y, 1)