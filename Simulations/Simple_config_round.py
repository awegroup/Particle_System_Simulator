# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:54:31 2024

@author: Mark
"""

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



class OpticalForceCalculatorCross():
    def __init__(self, PS, LB1, LB2):
        self.ParticleSystem = ParticleSystem
        self.PS = self.ParticleSystem #alias for convenience
        self.LaserBeam1 = LB1
        self.LaserBeam2 = LB2
        self.OFC1 = OpticalForceCalculator(PS, LB1)
        self.OFC2 = OpticalForceCalculator(PS, LB2)
        self.un_displace_particle_system = self.OFC1.un_displace_particle_system

    def force_value(self):
        return self.OFC1.force_value()+self.OFC2.force_value()

    def calculate_restoring_forces(self):
        f1, m1 = self.OFC1.calculate_restoring_forces()
        f2, m2 = self.OFC2.calculate_restoring_forces()
        return f1+f2, m1+m2

    def displace_particle_system(self, displacement, **kwargs):
        self.OFC1.displace_particle_system(displacement, **kwargs)

    def calculate_stability_coefficients(self, **kwargs):
        return self.OFC1.calculate_stability_coefficients(**kwargs) + self.OFC2.calculate_stability_coefficients(**kwargs)

    def plot(self,ax, **kwargs):
        LB1.plot(ax, **kwargs)
        LB2.plot(ax, **kwargs)

        return ax

def override_constraints(PS: ParticleSystem):
    for p in PS.particles:
        if p.fixed:
            if p.constraint_type == 'plane':
                p.set_fixed(False)



global_start_time = time.time()

# Setup parameters
params = {
    # model parameters
    "c": 1,  # [N s/m] damping coefficient
    "m_segment": 1, # [kg] mass of each node
    "thickness":200e-9, # [m] thickness of PhC
    "rho":3184, # [kg/m3]

    # simulation settings
    "dt": 3e-3,  # [s]       simulation timestep
    'adaptive_timestepping':2.5e-4, # [m] max distance traversed per timestep
    "t_steps": 1e3,  # [-]      max number of simulated time steps
    "abs_tol": 1e-20,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
    "max_iter": int(1e2),  # [-]       maximum number of iterations for the bicgstab solver

    # Simulation Steps
    "convergence_threshold": 5e-17,
    "min_iterations":30, # Should exceed the size of the force ringbuffer in the simulation loop

    # Mesh_dependent_settings
    "midstrip_width": 1,
    "boundary_margin": 0.175
    }

params['E'] = 470e9
params['G'] = 0
params['E_x'] = params['E']*7/100
params['E_y'] = params['E']*18/100

fill_factor = 0.40

params['rho'] *= fill_factor
density_ring = 2330 # [kg/m3] Support frame

# Setup mesh
n_segments = 20 # make sure this is uneven so there are no particles on the centerline
radius = 2.5e-3 #[m]
length = 2*radius
fixed_edge_width = radius/n_segments*2.1

mesh = MF.mesh_round_phc_square_cross(radius,
                                      mesh_edge_length=length/n_segments,
                                      params = params,
                                      noncompressive=True,
                                      fix_outer=True,
                                      edge = fixed_edge_width)
# params['k'] = params['k_y']

# mesh = MF.mesh_circle_square_cross(radius,
#                                    mesh_edge_length=length/n_segments,
#                                    params=params,
#                                    fix_outer = True,
#                                    edge = fixed_edge_width)

# We have to add some particles to act as a support structure.
stiffness_support = 3.52e4 # [n/m*m] line stiffness

n_fixed = sum([i[3] for i in mesh[1]])
circumference = 2 * np.pi * radius
k_support = stiffness_support * (circumference / n_fixed)
l_support = length/n_segments/10


multiplier = (radius+l_support)/radius

new_particles = []
for i, node in enumerate(mesh[1]):
    xyz = node[0].copy()
    if node[3]:
        node.append([0,0,1])
        node.append('plane')
        particle = [xyz*multiplier, np.zeros(3), params['m_segment'], True]
        link = [i, len(mesh[1])+len(new_particles), k_support, 1]
        new_particles.append(particle)
        mesh[0].append(link)

for p in new_particles:
    mesh[1].append(p)

# init particle system
PS = ParticleSystem(*mesh, params, clean_particles=True)
PS.calculate_correct_masses(params['thickness'], params['rho'])
starting_postions = PS.x_v_current_3D[0]

# %% Modifying PS
# Adding pre_stress
pre_stress = 300e6 # [Pa]
pre_strain = pre_stress / max(params['E_x'],params['E_y'])
shrink_factor = 1/(1+pre_strain)
PS.stress_self(shrink_factor)

#Adding dummy mass values for support:!
width_support = 25e-6 # [m]
m_support = width_support**2 * circumference*density_ring
m = sum([p.m for p in PS.particles])
n_fixed = sum([1 if p.fixed else 0 for p in PS.particles])
m_fixed = m_support/n_fixed
for p in PS.particles:
    if p.fixed:
        pass
        p.set_m(m_fixed)
m_new = sum([p.m for p in PS.particles])

#%% Setting up optical system
# import the photonic crystal(s)
dummy = interp.PhC_library['dummy']
gao = interp.PhC_library['Gao']
mark_4 = interp.PhC_library['Mark_4']
mark_5 = interp.PhC_library['Mark_5']


inner_phc = mark_5
inner_offset = np.pi
outer_phc = mark_4
outer_offset = 0

twist_compensation = -15/180*np.pi*0

r_transition = radius*3/5
#            phi_start,  phi_stop,   r_start,       r_stop,         midline,    PhC,        offset
regions1 = [[-np.pi/4,   np.pi/4,    0,             r_transition,   0,          inner_phc, inner_offset],
            [np.pi/4,    np.pi*3/4,  0,             r_transition,   np.pi*2/4,  inner_phc, inner_offset],
            [np.pi*3/4,  np.pi*4/4,  0,             r_transition,   np.pi*4/4,  inner_phc, inner_offset],
            [-np.pi*3/4, -np.pi*1/4, 0,             r_transition,   -np.pi*2/4, inner_phc, inner_offset],
            [-np.pi*4/4, -np.pi*3/4, 0,             r_transition,   -np.pi*4/4, inner_phc, inner_offset],
            [-np.pi/4,   np.pi/4,    r_transition,  length,         0,          outer_phc, outer_offset-twist_compensation],
            [np.pi/4,    np.pi*3/4,  r_transition,  length,         np.pi*2/4,  outer_phc, outer_offset+twist_compensation],
            [np.pi*3/4,  np.pi*4/4,  r_transition,  length,         np.pi*4/4,  outer_phc, outer_offset-twist_compensation],
            [-np.pi*3/4, -np.pi*1/4, r_transition,  length,         -np.pi*2/4, outer_phc, outer_offset+twist_compensation],
            [-np.pi*4/4, -np.pi*3/4, r_transition,  length,         -np.pi*4/4, outer_phc, outer_offset-twist_compensation]]

offset = np.pi*0
#            phi_start,  phi_stop,   r_start,    r_stop, midline
regions0 = [[0,          np.pi/2,    0,          length, np.pi*1/4, inner_phc, offset],
            [np.pi/2,    np.pi,      0,          length, np.pi*3/4, inner_phc, offset],
            [np.pi,      np.pi*3/2,  0,          length, np.pi*5/4, inner_phc, offset],
            [np.pi*3/2,  np.pi*2,    0,          length, np.pi*7/4, inner_phc, offset]]

regions = regions1
for reg in regions:
    reg[5] = interp.create_interpolator(reg[5], reg[4]+reg[6])

templog = []
for p in PS.particles:
    x,y,z = p.x
    phi = np.arctan2(y,x)
    r = np.linalg.norm(p.x)
    templog.append([phi])
    for reg in regions:
        if r >= reg[2] and r <= reg[3] and phi <= reg[1] and phi >= reg[0]:
            if reg[5]== ParticleOpticalPropertyType.SPECULAR:
                p.optical_type = ParticleOpticalPropertyType.SPECULAR
            else:
                p.optical_type = ParticleOpticalPropertyType.ARBITRARY_PHC
                p.optical_interpolator = reg[5]

templog= np.array(templog)
# init optical system
P = 400/2# [W] 400 divided by two because superposition of two orthogonally polarised beams
P /= 4 # There is a factor four difference in the force that I cannot manage to explain. soz.
# It works fine in different sims, just not here.
# if you want to check, set it all to specular and set sigma to radius/3
# net force should be P_original/c*2*2 (*2 for reflection, *2 for the second laser)
mu_x = 0
mu_y = 0
sigma = radius*2
I_0 = 2*P / (np.pi* sigma**2)


# Setting up two beams for cross-polarisation purposes
LB1 = LaserBeam(lambda x, y: I_0 * np.exp(-1/2 *((x-mu_x)/sigma)**2 # gaussian laser
                                          -1/2 *((y-mu_y)/sigma)**2),
                lambda x,y: np.outer(np.ones(x.shape),[0,1]))
LB2 = LaserBeam(lambda x, y: I_0 * np.exp(-1/2 *((x-mu_x)/sigma)**2 # gaussian laser
                                          -1/2 *((y-mu_y)/sigma)**2),
                lambda x,y: np.outer(np.ones(x.shape),[1,0]))

OFC = OpticalForceCalculatorCross(PS, LB1, LB2)

# pick the desired simulation
SIM = Simulate_Lightsail(PS,OFC,params)

#%% Start of use
# Run a cheeky little simulation?
cheeky = False
if cheeky:
    SIM.run_simulation(plotframes=0, printframes=10, simulation_function='kinetic_damping',file_id='_simple_')
    fig_convergence = plt.figure()
    ax_kin = fig_convergence.add_subplot(211)
    ax_kin.semilogy(PS.history["E_kin"])
    ax_kin.set_ylabel('E_kin')
    ax_f = fig_convergence.add_subplot(212)
    ax_f.semilogy(PS.history["net_force"])
    ax_f.set_ylabel('net_force')

    f = OFC.force_value()
    f_net = np.sum(f, axis=0)
    f_abs = np.linalg.norm(f_net)
    ax = PS.plot_forces(f, length = 1/f_abs/10)
    ax.set_aspect('auto')

    f_react = PS.find_reaction_forces()
    f_r_net = np.linalg.norm(f_react[:,:2],axis=1) #eliminate z force for now
    f_circ = np.sum(f_r_net)/circumference # [N/m]
    print(f'The force in the boundary is {f_circ:.4f} [N/m]')


# Vis laser and beam
beam_plot = False
if beam_plot:
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax = OFC.plot(ax,x_range = (-3*sigma,3*sigma), y_range = (-3*sigma,3*sigma),
                  z_scale = I_0/radius, arrow_length = radius/15, number_of_points = 18**2)
    PS.plot(ax)


stability_check = False
if stability_check:
    J = OFC.calculate_stability_coefficients(displacement_range = [0.1*radius, 3])
    print(J)

force_plot = True
force_check = True
if force_check:
    for i in range(5):
        f = OFC.force_value()
        PS.simulate(f.ravel())

    f = OFC.force_value()
    f_lift = m_new*9.80665
    f_z = np.sum(f, axis=0)[-1]
    print(f'req. {f_lift}, avail. {f_z}')

    disp_list = [[0,0,0,0,0,0],
                [0,0,0,3,0,0],
                [0,0,0,0,3,0],
                [0.1*radius,0,0,0,0,0],
                [0,0.1*radius,0,0,0,0]]#,
                # [0,0,0,0,0,3],
                # [0,0,0,0,0,-3]]
    for disp in disp_list:
        OFC.displace_particle_system(disp,suppress_warnings=True)
        f = OFC.force_value()
        #f = np.nan_to_num(f)
        f_net = np.sum(f, axis=0)
        f_abs = np.linalg.norm(f_net)
        if force_plot:
            PS.plot_forces(f, length = 1/f_abs/2)
        f_res,m_res =OFC.calculate_restoring_forces()
        print(disp, f_res, m_res)
        OFC.un_displace_particle_system()


trajectory = False
params["t_steps"]= 1e4
PS.COM_offset = np.array([0,0,-width_support/2])
if trajectory:
    override_constraints(PS)
    SIM.simulate_trajectory(plotframes=1e5,
                            printframes=10,
                            plot_forces=True,
                            file_id = '_check_5_',
                            deform = False,
                            rotate = False)

