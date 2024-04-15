# -*- coding: utf-8 -*-
"""
It's dirty because I just pasted the contents of the other file in this function and started hacking
away at the parts I didn't need. Now everytime I make a change I have to have both files side-by-side
and make sure I retain parity so I can use the normal one to do tests for the sweep -_-'
"""
#%% Setup
import time
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import c

from src.particleSystem.ParticleSystem import ParticleSystem
from src.Sim.simulations import Simulate_Lightsail
import src.Mesh.mesh_functions as MF
import src.ExternalForces.optical_interpolators.interpolators as interp
from src.ExternalForces.LaserBeam import LaserBeam
from src.ExternalForces.OpticalForceCalculator import OpticalForceCalculator
from src.ExternalForces.OpticalForceCalculator import ParticleOpticalPropertyType

def sweep_gao(stiffness_support = 0.1): # [n/m*m] line stiffness
    intro_msg = f'Starting sweep for {stiffness_support=}'
    print('\n')
    print('='*len(intro_msg))
    print(intro_msg)
    print('-'*len(intro_msg))
    print('\n')
    global_start_time = time.time()

    # Setup parameters
    params = {
        # model parameters
        "c": 1,  # [N s/m] damping coefficient
        "m_segment": 1, # [kg] mass of each node
        "thickness":100e-9, # [m] thickness of PhC

        # simulation settings
        "dt": 0.1,  # [s]       simulation timestep
        'adaptive_timestepping':1e-2, # [m] max distance traversed per timestep
        "t_steps": 1e6,  # [-]      max number of simulated time steps
        "abs_tol": 1e-10,  # [m/s]     absolute error tolerance iterative solver
        "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
        "max_iter": int(1e2),  # [-]       maximum number of iterations for the bicgstab solver

        # Simulation Steps
        "convergence_threshold": 1e-6,
        "min_iterations":30,

        # Mesh_dependent_settings
        "midstrip_width": 1,
        "boundary_margin": 0.175
        }

    params['E'] = 470e9
    params['G'] = 0
    params['E_x'] = params['E']*7/100
    params['E_y'] = params['E']*18/100


    # Setup mesh
    n_segments = 25 # make sure this is uneven so there are no particles on the centerline
    length = 1
    mesh = MF.mesh_phc_square_cross(length,
                                    mesh_edge_length=length/n_segments,
                                    params = params,
                                    noncompressive=True)
    # We have to add some particles to act as a support structure.
    k_support = stiffness_support / (length / n_segments)
    l_support = length/n_segments/50

    simulate_3D = False

    for i in range((n_segments+1)**2):
        # calculate coordinates
        xyz = mesh[1][i][0].copy()
        if xyz[1] ==0 and simulate_3D:
            xyz[1]-=l_support
        elif xyz[1] == length and simulate_3D:
            xyz[1]+=l_support
        elif xyz[0] == 0:
            xyz[0]-=l_support
        elif xyz[0] == length:
            xyz[0]+=l_support


        if np.any(xyz != mesh[1][i][0]):
            particle = [xyz, np.zeros(3), params['m_segment'], True]
            link = [i, len(mesh[1]), k_support, 1]
            mesh[1].append(particle)
            mesh[0].append(link)

        xyz = mesh[1][i][0].copy()
        if (np.all(xyz == [0,0,0])
            or np.all(xyz == [0,length,0])
            or np.all(xyz == [length,0,0])
            or np.all(xyz == [length,length,0])) and simulate_3D:

            if xyz[0] ==0:
                xyz[0]-=l_support
            elif xyz[0] == length:
                xyz[0]+=l_support

            if np.any(xyz != mesh[1][i][0]):
                particle = [xyz, np.zeros(3), params['m_segment'], True]
                link = [i, len(mesh[1]), k_support, 1]
                mesh[1].append(particle)
                mesh[0].append(link)

    # init particle system
    PS = ParticleSystem(*mesh, params, clean_particles=False)

    # Setup the optical sytem
    I_0 = 100e9 /(10*10)
    mu_x = 0.5
    mu_y = 0.5
    sigma = 1/2
    w=2*length
    if simulate_3D:
        LB = LaserBeam(lambda x, y: I_0 * np.exp(-1/2 *((x-mu_x)/sigma)**2 # gaussian laser
                                                 -1/2 *((y-mu_y)/sigma)**2),
                       lambda x,y: np.outer(np.ones(x.shape),[0,1]))
    else:
        LB = LaserBeam(lambda x, y: I_0 * np.exp(-2*((x-mu_x)/w)**2), # gaussian laser
                       lambda x,y: np.outer(np.ones(x.shape),[0,1]))
    # Import the crystal
    fname = interp.PhC_library['Gao']
    #fname = interp.PhC_library['dummy']
    interp_right_side = interp.create_interpolator(fname,np.pi)
    interp_left_side = interp.create_interpolator(fname, 0)


    # set the correct boundary conditions and crystals on the particle system
    for p in PS.particles:
        if simulate_3D:
            if p.x[1] == 0 or p.x[1] == length:
                p.set_fixed(True, [0,0,1], 'plane')

        if p.x[0] == 0 or p.x[0] == length:
            p.set_fixed(True, [0,0,1], 'plane')

        p.optical_type = ParticleOpticalPropertyType.ARBITRARY_PHC
        if p.x[0]>length/2:
            p.optical_interpolator = interp_right_side
        else:
            p.optical_interpolator = interp_left_side

    OFC = OpticalForceCalculator(PS, LB)
    SIM = Simulate_Lightsail(PS,OFC,params)

    #%% Getting translation and rotation data
    printframes = 50
    n_points = 17
    print('Starting rotations and translations')
    translations = np.linspace(-length,length,n_points)
    rotations = np.linspace(-10,10,n_points)

    translation_plot = []

    resimulate_on_displacement = True
    print("\n\nCalculating translation")
    for i, t in enumerate(translations):
        print(f'\nTranslation {t=}')
        OFC.displace_particle_system([t,0,0,0,0,0])

        if resimulate_on_displacement:
            SIM.run_simulation(plotframes=0,
                               printframes=printframes,
                               simulation_function='kinetic_damping',
                               file_id=f'_{t}_')
        net_force, net_moments = OFC.calculate_restoring_forces()

        # The force data is a little sensitive to random fluctuation, so instead I'm pulling the last
        # 50 entries from a ring buffer I added to the history.
        net_force = np.array([np.sum(forces,axis=0) for forces in PS.history['forces_ringbuffer']])
        net_force = np.sum(net_force, axis=0)/len(PS.history['forces_ringbuffer'])

        OFC.un_displace_particle_system()

        translation_plot.append([t, *net_force, *net_moments])

    rotation_plot=[]

    print("\n\nCalculating rotations")
    for i, r in enumerate(rotations):
        OFC.displace_particle_system([0,0,0,0,r,0])
        print(f'\nRotation {r=}')
        if resimulate_on_displacement:
            SIM.run_simulation(plotframes=0,
                               printframes=printframes,
                               simulation_function='kinetic_damping',
                               file_id=f'_{r}_')
        net_force, net_moments = OFC.calculate_restoring_forces()

        # The force data is a little sensitive to random fluctuation, so instead I'm pulling the last
        # 30 entries from a ring buffer I added to the history.
        net_force = np.array([np.sum(forces,axis=0) for forces in PS.history['forces_ringbuffer']])
        net_force = np.sum(net_force, axis=0)/len(PS.history['forces_ringbuffer'])

        OFC.un_displace_particle_system()

        rotation_plot.append([r, *net_force, *net_moments])


    translation_plot= np.array(translation_plot)
    rotation_plot = np.array(rotation_plot)
    header = ['displacement', 'Fx', 'Fy', 'Fz', 'Rx', 'Ry', 'Rz']
    pd.DataFrame(translation_plot,columns=header).to_csv(f"temp/translation_{stiffness_support=}.csv", header=True, index = False)
    pd.DataFrame(rotation_plot,columns=header).to_csv(f"temp/rotation_{stiffness_support=}.csv", header=True, index = False)

    delta_time = time.time()-global_start_time
    print(f'All in all that took {delta_time//60:.0f}m {delta_time%60:.2f}s')

    # %% Main block
if __name__ == '__main__':
    #sweep_list = [1e3,1e2,1e1,1,0.1]
    sweep_list = [0.25,0.5,0.75,2.5,5,7.5][::-1]
    for k in sweep_list:
        sweep_gao(k)