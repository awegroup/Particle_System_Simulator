# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:54:31 2024

@author: Mark
"""

import time
import logging
import copy
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.constants import c
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

from PSS.particleSystem.ParticleSystem import ParticleSystem
from PSS.Sim.simulations import Simulate_Lightsail
import PSS.Mesh.mesh_functions as MF
import PSS.ExternalForces.optical_interpolators.interpolators as interp
from PSS.ExternalForces.LaserBeam import LaserBeam
from PSS.ExternalForces.OpticalForceCalculator import (
    OpticalForceCalculator,
)
from PSS.ExternalForces.OpticalForceCalculator import (
    ParticleOpticalPropertyType,
)


class OpticalForceCalculatorCross:
    def __init__(self, PS, LB1, LB2):
        self.ParticleSystem = ParticleSystem
        self.PS = self.ParticleSystem  # alias for convenience
        self.LaserBeam1 = LB1
        self.LaserBeam2 = LB2
        self.OFC1 = OpticalForceCalculator(PS, LB1)
        self.OFC2 = OpticalForceCalculator(PS, LB2)

    def force_value(self):
        return self.OFC1.force_value() + self.OFC2.force_value()

    def calculate_restoring_forces(self, **kwargs):
        f1, m1 = self.OFC1.calculate_restoring_forces(**kwargs)
        f2, m2 = self.OFC2.calculate_restoring_forces(**kwargs)
        return f1 + f2, m1 + m2

    def calculate_stability_coefficients(self, **kwargs):
        return self.OFC1.calculate_stability_coefficients(
            **kwargs
        ) + self.OFC2.calculate_stability_coefficients(**kwargs)

    def plot(self, ax, **kwargs):
        LB1.plot(ax, **kwargs)
        LB2.plot(ax, **kwargs)

        return ax


def override_constraints(PS: ParticleSystem):
    for p in PS.particles:
        if p.fixed:
            if p.constraint_type == "plane":
                p.set_fixed(False)


global_start_time = time.time()

# Setup parameters
params = {
    # model parameters
    "c": 1,  # [N s/m] damping coefficient
    "m_segment": 1,  # [kg] mass of each node
    "thickness": 200e-9,  # [m] thickness of PhC
    "rho": 3184,  # [kg/m3]
    # simulation settings
    "dt": 2e-3,  # [s]       simulation timestep
    "adaptive_timestepping": 2.5e-4,  # [m] max distance traversed per timestep
    "t_steps": 1e3,  # [-]      max number of simulated time steps
    "abs_tol": 1e-20,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
    "max_iter": int(
        1e2
    ),  # [-]       maximum number of iterations for the bicgstab solver
    # Simulation Steps
    "convergence_threshold": 5e-8,  # Metric depends on size of timestep. Have to update them together.
    "min_iterations": 30,  # Should exceed the size of the force ringbuffer in the simulation loop
    # Mesh_dependent_settings
    "midstrip_width": 1,
    "boundary_margin": 0.175,
}

params["E"] = 470e9
params["G"] = 0
params["E_x"] = params["E"] * (259 + 351) / 1991
params["E_y"] = params["E"] * 5 / 100
E_av = (params["E_x"] + params["E_y"]) / 2

fill_factor = 0.43

params["rho"] *= fill_factor
density_ring = 2330  # [kg/m3] Support frame

# Setup mesh
n_segments = 25  # make sure this is uneven so there are no particles on the centerline
radius = 0.5e-3  # [m]
length = 2 * radius
fixed_edge_width = radius / n_segments * 1.975

mesh = MF.mesh_round_phc_square_cross(
    radius,
    mesh_edge_length=length / n_segments,
    params=params,
    noncompressive=True,
    fix_outer=True,
    edge=fixed_edge_width,
)

# Method for detecting edge particles isn't great, so we're overiding to add the ones it missed
link_counting = defaultdict(int)
for link in mesh[0]:
    link_counting[link[0]] += 1
    link_counting[link[1]] += 1

for particle_i in link_counting.keys():
    if link_counting[particle_i] < 4:
        mesh[1][particle_i][3] = True

# We have to add some particles to act as a support structure.
stiffness_support = 5.93e07  # [n/m*m] line stiffness

n_fixed = sum([i[3] for i in mesh[1]])
circumference = 2 * np.pi * radius
k_support = stiffness_support * (circumference / n_fixed)
l_support = length / n_segments / 10

multiplier = (radius + l_support) / radius


new_particles = []
for i, node in enumerate(mesh[1]):
    xyz = node[0].copy()
    if node[3]:
        node.append([0, 0, 1])
        node.append("plane")
        particle = [xyz * multiplier, np.zeros(3), params["m_segment"], True]
        link = [i, len(mesh[1]) + len(new_particles), k_support, 1]
        new_particles.append(particle)
        mesh[0].append(link)

for p in new_particles:
    mesh[1].append(p)

# init particle system
PS = ParticleSystem(*mesh, params, clean_particles=True)
PS.calculate_correct_masses(params["thickness"], params["rho"])
starting_postions = PS.x_v_current_3D[0]

# Rotate PS around z so there aren't any particles on region boundaries
PS.displace([0, 0, 0, 0, 0, 45])
# and deal with a slight meshing assymetry
pos_offset = [-0.00493598 * length, 0.00474392 * length]
# PS.displace([*pos_offset, 0, 0, 0, 0], suppress_warnings= True) #!!! restore me
PS.current_displacement = None
# %% Modifying PS
# Adding pre_stress
pre_stress = 300e6  # [Pa]
pre_strain = pre_stress / max(params["E_x"], params["E_y"])
shrink_factor = 1 / (1 + pre_strain)
PS.stress_self(shrink_factor)

# Adding dummy mass values for support:!
width_support = 17e-6  # [m]
mean_circ_ring = (radius - width_support / 2) * np.pi * 2
m_support = width_support**2 * mean_circ_ring * density_ring
m = sum([p.m for p in PS.particles])
n_fixed = sum([1 if p.fixed else 0 for p in PS.particles])
m_fixed = m_support / n_fixed
for p in PS.particles:
    if p.fixed:
        p.set_m(m_fixed)
m_new = sum([p.m for p in PS.particles])

# %% Setting up optical system
# import the photonic crystal(s)
dummy = interp.PhC_library["dummy"]
gao = interp.PhC_library["Gao"]
mark_4 = interp.PhC_library["Mark_4"]  # likes offset 0
mark_4_1 = interp.PhC_library["Mark_4.1"]  # likes offset 0
mark_5 = interp.PhC_library["Mark_5"]
mark_6 = interp.PhC_library["Mark_6"]  # likes offset pi
mark_7 = interp.PhC_library["Mark_7"]  # likes offset 0
mark_8 = interp.PhC_library["Mark_8"]  # likes offset pi
mark_9 = interp.PhC_library["Mark_9"]  # likes offset pi

inner_phc = mark_6
inner_offset = np.pi
outer_phc = mark_4
outer_offset = np.pi * 0

twist_compensation = 0 / 180 * np.pi

r_transition = radius * 0 / 5
#            phi_start,  phi_stop,   r_start,       r_stop,         midline,    PhC,        offset
regions1 = [
    [-np.pi / 4, np.pi / 4, 0, r_transition, 0, inner_phc, inner_offset],
    [np.pi / 4, np.pi * 3 / 4, 0, r_transition, np.pi * 2 / 4, inner_phc, inner_offset],
    [
        np.pi * 3 / 4,
        np.pi * 4 / 4,
        0,
        r_transition,
        np.pi * 4 / 4,
        inner_phc,
        inner_offset,
    ],
    [
        -np.pi * 3 / 4,
        -np.pi * 1 / 4,
        0,
        r_transition,
        -np.pi * 2 / 4,
        inner_phc,
        inner_offset,
    ],
    [
        -np.pi * 4 / 4,
        -np.pi * 3 / 4,
        0,
        r_transition,
        -np.pi * 4 / 4,
        inner_phc,
        inner_offset,
    ],
    [
        -np.pi / 4,
        np.pi / 4,
        r_transition,
        length,
        0,
        outer_phc,
        outer_offset - twist_compensation,
    ],
    [
        np.pi / 4,
        np.pi * 3 / 4,
        r_transition,
        length,
        np.pi * 2 / 4,
        outer_phc,
        outer_offset + twist_compensation,
    ],
    [
        np.pi * 3 / 4,
        np.pi * 4 / 4,
        r_transition,
        length,
        np.pi * 4 / 4,
        outer_phc,
        outer_offset - twist_compensation,
    ],
    [
        -np.pi * 3 / 4,
        -np.pi * 1 / 4,
        r_transition,
        length,
        -np.pi * 2 / 4,
        outer_phc,
        outer_offset + twist_compensation,
    ],
    [
        -np.pi * 4 / 4,
        -np.pi * 3 / 4,
        r_transition,
        length,
        -np.pi * 4 / 4,
        outer_phc,
        outer_offset - twist_compensation,
    ],
]

offset = np.pi * 0
inner_phc = mark_6
#            phi_start,  phi_stop,   r_start,    r_stop, midline
regions0 = [
    [0, np.pi / 2, 0, length, np.pi * 1 / 4, inner_phc, offset],
    [np.pi / 2, np.pi, 0, length, np.pi * 3 / 4, inner_phc, offset],
    [np.pi, np.pi * 3 / 2, 0, length, np.pi * 5 / 4, inner_phc, offset],
    [np.pi * 3 / 2, np.pi * 2, 0, length, np.pi * 7 / 4, inner_phc, offset],
]

regions = regions1
for reg in regions:
    if type(reg[5]) == str:
        reg[5] = interp.create_interpolator(reg[5], reg[4] + reg[6])

templog = []
for p in PS.particles:
    x, y, z = p.x
    phi = np.arctan2(y, x)
    r = np.linalg.norm(p.x)
    templog.append([phi])
    for reg in regions:
        if r >= reg[2] and r <= reg[3] and phi <= reg[1] and phi >= reg[0]:
            if reg[5] == ParticleOpticalPropertyType.SPECULAR:
                p.optical_type = ParticleOpticalPropertyType.SPECULAR
            else:
                p.optical_type = ParticleOpticalPropertyType.ARBITRARY_PHC
                p.optical_interpolator = reg[5]

specular_override = False
if specular_override:
    for p in PS.particles:
        p.optical_type = ParticleOpticalPropertyType.SPECULAR

# now let's fix the rims
# Nvm this breaks everything.
# dummy = interp.create_interpolator(dummy)
# for p in PS.particles:
#     if p.constraint_type == 'point':
#         p.optical_interpolator = dummy

templog = np.array(templog)
# init optical system
P = (
    400 / 2
)  # [W] 400 divided by two because superposition of two orthogonally polarised beams

# if you want to check, set it all to specular and set sigma to radius/3
# net force should be P_original/c*2*2 (*2 for reflection, *2 for the second laser)
mu_x = 0
mu_y = 0
sigma = radius * 5.5
I_0 = 2 * P / (np.pi * sigma**2)


# Setting up two beams for cross-polarisation purposes
LB1 = LaserBeam(
    lambda x, y: I_0
    * np.exp(
        -1 / 2 * ((x - mu_x) / sigma) ** 2  # gaussian laser
        - 1 / 2 * ((y - mu_y) / sigma) ** 2
    ),
    lambda x, y: np.outer(np.ones(x.shape), [0, 1]),
)
LB2 = LaserBeam(
    lambda x, y: I_0
    * np.exp(
        -1 / 2 * ((x - mu_x) / sigma) ** 2  # gaussian laser
        - 1 / 2 * ((y - mu_y) / sigma) ** 2
    ),
    lambda x, y: np.outer(np.ones(x.shape), [1, 0]),
)

OFC = OpticalForceCalculatorCross(PS, LB1, LB2)

# pick the desired simulation
SIM = Simulate_Lightsail(PS, OFC, params)

# %% Start of using and plotting
# %%%  Run a cheeky little simulation?
cheeky = False
if cheeky:
    PS.params["convergence_threshold"] = 1e-20

    SIM.run_simulation(
        plotframes=0,
        printframes=10,
        simulation_function="kinetic_damping",
        file_id="_simple_",
    )
    fig_convergence = plt.figure()
    ax_kin = fig_convergence.add_subplot(211)
    ax_kin.semilogy(PS.history["E_kin"])
    ax_kin.set_ylabel("E_kin")
    ax_f = fig_convergence.add_subplot(212)
    ax_f.semilogy(PS.history["net_force"])
    ax_f.set_ylabel("net_force")

    f = OFC.force_value()
    f_net = np.sum(f, axis=0)
    f_abs = np.linalg.norm(f_net)
    zmax = PS.x_v_current_3D[0][:, 2].max()
    ax = PS.plot_forces(f, length=zmax * 2e9)
    ax.set_aspect("auto")
    zmax /= 1e6
    ax.set_zlim([-zmax / 2, zmax])

    f_react = PS.find_reaction_forces()
    f_r_net = np.linalg.norm(f_react[:, :2], axis=1)  # eliminate z force for now
    f_circ = np.sum(f_r_net) / circumference  # [N/m]
    print(
        f"Pre-stress {pre_stress/1e6} [MPa], Edge stiffness {stiffness_support/1e3:.4g} [kN/m], Boundary Load {f_circ:.4f} [N/m]"
    )

    f_lift = m_new * 9.80665
    f_z = np.sum(f, axis=0)[-1]
    print(f"req. {f_lift}, avail. {f_z}")


# %%% Vis laser and sail
beam_plot = False
if beam_plot:
    fig1 = plt.figure(figsize=(12, 6))
    ax = fig1.add_subplot(projection="3d")
    ax = OFC.plot(
        ax,
        x_range=(-3 * sigma, 3 * sigma),
        y_range=(-3 * sigma, 3 * sigma),
        z_scale=I_0 / radius,
        arrow_length=radius / 15,
        number_of_points=25**2,
    )
    PS.displace([radius, 0, 1e-3, 0, 0, 0])
    PS.plot_forces(OFC.force_value(), ax, length=1e7)
    ax.azim = 0
    ax.elev = 0
    fig1.tight_layout()
    ax.set_proj_type("ortho")

# %%% Jacobian printing
stability_check = False
if stability_check:
    J = OFC.calculate_stability_coefficients(displacement_range=[0.1 * radius, 0.5])
    J[:, 2] = 0
    J[:, -1] = 0
    J[2, :] = 0
    J[-1, :] = 0
    J_no_z = J[J != 0].reshape([4, 4])
    print(J)
    print(np.linalg.det(J_no_z))

# %%% Force vector plots with displacements
force_plot = False
force_check = False
if force_check:
    for i in range(100):
        f = OFC.force_value()
        PS.simulate(f.ravel())

    f = OFC.force_value()
    f_lift = m_new * 9.80665
    f_z = np.sum(f, axis=0)[-1]
    print(f"req. {f_lift}, avail. {f_z}")

    disp_list = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 3, 0],
        [radius / 10, 0, 0, 0, 0, 0],
        [0, radius / 10, 0, 0, 0, 0],
    ]  # ,
    # [0,radius/3,0,3,0,0]]#,
    # [0,0,0,0,0,5],
    # [0,0,0,0,0,-5]]
    for disp in disp_list:
        PS.displace(disp, suppress_warnings=True)
        f = OFC.force_value()
        # f = np.nan_to_num(f)
        f_net = np.sum(f, axis=0)
        f_abs = np.linalg.norm(f_net)
        if force_plot:
            ax = PS.plot_forces(f, length=1 / f_abs / 6)
            ax.figure.tight_layout()
        f_res, m_res = OFC.calculate_restoring_forces()
        padding = 25 - len(str(disp))
        print(
            disp,
            " " * padding,
            "f_res",
            *[f"\t{i:.3g}" for i in f_res],
            "\tm_res",
            *[f"\t{i:.3g}" for i in m_res],
        )
        PS.un_displace()


# %%% Trajectory plots
params["t_steps"] = 1e5
PS.COM_offset = np.array([0, 0, -width_support / 2])

#            Pressure Damping coefficient
drag_data_1 = {
    101325: 8.1652e-06,
    80000: 6.4288e-06,
    60000: 4.8406e-06,
    40000: 3.2501e-06,
    20000: 1.6571e-06,
    10000: 8.5704e-07,
    1000: 1.2375e-07,
}

drag_data_2 = {
    101325: 2.0916e-06,
    80000: 1.6505e-06,
    60000: 1.2477e-06,
    40000: 8.429e-07,
    20000: 4.314e-07,
    10000: 2.326e-07,
    1000: 4.174e-08,
}

# Define damping coeffs, make sure they're negative!
damping_pressure = 10000
lin_damping = -drag_data_2[damping_pressure]
rot_damping = -1e-16
damping = np.array([lin_damping, lin_damping, 0, rot_damping, rot_damping, rot_damping])

trajectory = False
if trajectory:
    f = OFC.force_value()
    f_lift = m_new * 9.80665
    f_z = np.sum(f, axis=0)[-1]
    print(f"req. {f_lift}, avail. {f_z}")

    # PS.displace([0,radius/10,0,0,0,0],suppress_warnings=True)
    initial_conditions = np.array(
        [[radius * 0.02, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float64
    )
    override_constraints(PS)
    SIM.simulate_trajectory(
        plotframes=0,
        printframes=100,
        plot_forces=False,
        plot_net_force=True,
        plot_angles=[10, 0],
        file_id="_M9_damping_",
        deform=True,
        gravity=True,
        # initial_conditions=initial_conditions,
        damping=damping,
        spin=False,
    )
    z = (np.cumsum(PS.history["position"], axis=0)[-1] / length)[2]
    print(f"Final altitude is {z}")
    SIM.plot_flight_hist(pos_offset=pos_offset)


# %%% Trajectory-sweep from initial conditions plot
def trajectory_from_initial_conditions(x0=0, y0=0, theta_x0=0, theta_y0=0):
    override_constraints(PS)
    # Reset PS
    PS.update_pos_unsafe(PS.initial_positions)
    v = np.zeros(PS.n * 3)
    PS.update_vel_unsafe(v)
    PS.reset_history()

    # Perform initial displacement
    PS.displace([x0, y0, 0, theta_x0, theta_y0, 0], suppress_warnings=True)
    PS.params["convergence_threshold"] = radius * 2e-2

    stable = SIM.simulate_trajectory(
        plotframes=0,
        printframes=100,
        plot_forces=False,
        plot_net_force=True,
        plot_angles=[10, 0],
        file_id="_M4_damping_",
        deform=False,
        spin=False,
        gravity=True,
        damping=damping,
    )

    pos = np.sum(PS.history["position"], axis=0) / length
    print(f"Final altitude is {pos[2]}")
    return stable, pos


trajectory_sweep = False
if trajectory_sweep:
    PS.initial_positions, _ = PS.x_v_current
    start = time.time()
    intro = "Running trajectory sweep"
    buffer = int((80 - len(intro) - 1) / 2)
    print(80 * "=" + "\n" + buffer * " " + intro + "\n" + 80 * "-")
    x0 = np.linspace(0, 1, 31) ** 3 / 8 * radius
    y0 = np.linspace(0, 1, 31) ** 3 / 8 * radius
    x0, y0 = np.meshgrid(x0, y0)
    results = []

    space = 62
    for i, initials in enumerate(zip(x0.ravel(), y0.ravel())):
        progress = round(i / len(x0.ravel()), 3)
        equals = round(progress * space)
        if progress < 0.1:
            offset = 1
        else:
            offset = 0
        print(
            f"Progress: {progress*100:.1f}% "
            + " " * offset
            + "["
            + "=" * equals
            + " " * (space - equals)
            + "]"
        )
        print(f"Initial conditions: {initials=}")
        stable, pos = trajectory_from_initial_conditions(*initials)
        r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        converged = r < 1
        results.append((stable, pos))  # , PS.history))
        print(80 * "-")

    current = time.time()
    elapsed = current - start
    el_m = round(elapsed / 60)
    el_s = elapsed % 60
    print(f"Done! that took {el_m}m {el_s}s")
    print(80 * "=")

    stable = np.array([i[0] for i in results])
    altitude = np.array([i[1][2] for i in results])

    stable = stable.reshape(x0.shape)
    altitude = altitude.reshape(x0.shape)

    marking = round(time.time())
    np.savetxt(
        f"temp/initial_pos_sweep/sweep_trajectories_stable_{marking}.csv", stable
    )
    np.savetxt(f"temp/initial_pos_sweep/sweep_trajectories_x_{marking}.csv", x0)
    np.savetxt(f"temp/initial_pos_sweep/sweep_trajectories_y_{marking}.csv", y0)

    altitude[stable] = 0

    stability_contours = plt.figure(figsize=[16, 6])
    ax1 = stability_contours.add_subplot(122)
    contour = ax1.contourf(
        x0 / length, y0 / length, altitude, cmap="inferno"
    )  # , levels = 3)
    cbar = stability_contours.colorbar(contour)
    cbar.set_label("Altitude [D]")
    plt.xlabel("$x_0$ [D]")
    plt.ylabel("$y_0$ [D]")
    ax1.set_aspect("equal")

    ax2 = stability_contours.add_subplot(121)
    stab_region = ax2.contourf(
        x0 / length, y0 / length, stable, cmap="binary", levels=1
    )
    cbar = stability_contours.colorbar(stab_region)
    cbar.set_label("Stability (1 = stable)")
    plt.xlabel("$x_0$ [D]")
    plt.ylabel("$y_0$ [D]")
    ax2.set_aspect("equal")

    stability_contours.tight_layout()

    stable_rot90 = np.rot90(stable, k=3)
    stable_rot180 = np.rot90(stable, k=2)
    stable_rot270 = np.rot90(stable, k=1)

    # Tile the image
    top = np.concatenate((stable_rot180, stable_rot270), axis=1)
    bottom = np.concatenate((stable_rot90, stable), axis=1)
    full_domain = np.concatenate((top, bottom), axis=0)

    extent = [
        -np.max(x0) / length,
        np.max(x0) / length,
        -np.max(y0) / length,
        np.max(y0) / length,
    ]

    fig2 = plt.figure()
    ax_x = fig2.add_subplot()
    im = ax_x.imshow(full_domain, cmap="binary", origin="lower", extent=extent)
    cbar2 = fig2.colorbar(im)
    cbar2.set_label("Stability (1 = stable)")
    ax_x.set_xlabel("$x_0$ [D]")
    ax_x.set_ylabel("$y_0$ [D]")


# %%% Displacement Reaction plots
def collect_reaction_data(PS, OFC, disp_range, disp_type):
    reactions = []
    base_range = np.linspace(-1, 1, 101)
    modified_range = abs(base_range) ** (1 / 2) * np.sign(
        base_range
    )  # concentrates values around 0
    displacements = modified_range * disp_range
    rotational = True if "rot" in disp_type else False

    for disp in displacements:
        if disp_type in ["trans_x", "trans_y", "trans_z"]:
            displacement = [
                disp if disp_type == f"trans_{axis}" else 0 for axis in ["x", "y", "z"]
            ]
            displacement += [0, 0, 0]
        elif disp_type in ["rot_x", "rot_y", "rot_z"]:
            displacement = [0, 0, 0]
            if rotational:
                displacement += [
                    disp if disp_type == f"rot_{axis}" else 0
                    for axis in ["x", "y", "z"]
                ]
            else:
                displacement += [0, 0, 0]
        else:
            raise ValueError("Invalid displacement type")

        PS.displace(displacement, suppress_warnings=True)
        f = OFC.force_value()
        f_res, m_res = OFC.calculate_restoring_forces(forces=f)
        reaction = np.concatenate((f_res, m_res))
        reactions.append(reaction)
        # PS.plot_forces(f, length = 1e6)
        PS.un_displace()

    return displacements, np.array(reactions)


def plot_displacement_vs_reaction(PS, OFC, disp_range_trans, disp_range_rot):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    results = []

    # Translation vs. Vertical Force
    displacements, reactions = collect_reaction_data(
        PS, OFC, disp_range_trans, "trans_x"
    )
    results.append((displacements, reactions))
    axs[0, 0].plot(displacements / length, reactions[:, 2], label="Vertical Force")
    axs[0, 0].set_xlabel("Displacement [D]")
    axs[0, 0].set_ylabel("Vertical Force [N]")
    axs[0, 0].set_title("Translation vs. Vertical Force")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Translation vs. Lateral Force
    axs[0, 1].plot(displacements / length, reactions[:, 0], label="Lateral Force X")
    axs[0, 1].plot(displacements / length, reactions[:, 1], label="Lateral Force Y")
    axs[0, 1].set_xlabel("Displacement [D]")
    axs[0, 1].set_ylabel("Lateral Force [N]")
    axs[0, 1].set_title("Translation vs. Lateral Force")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Translation vs. Tipping Moment
    axs[0, 2].plot(displacements / length, reactions[:, 3], label="Tipping Moment X")
    axs[0, 2].plot(displacements / length, reactions[:, 4], label="Tipping Moment Y")
    axs[0, 2].set_xlabel("Displacement [D]")
    axs[0, 2].set_ylabel("Tipping Moment [N·m]")
    axs[0, 2].set_title("Translation vs. Tipping Moment")
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Rotation vs. Vertical Force
    displacements, reactions = collect_reaction_data(PS, OFC, disp_range_rot, "rot_y")
    results.append((displacements, reactions))
    axs[1, 0].plot(displacements, reactions[:, 2], label="Vertical Force")
    axs[1, 0].set_xlabel("Displacement [deg]")
    axs[1, 0].set_ylabel("Vertical Force [N]")
    axs[1, 0].set_title("Rotation vs. Vertical Force")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Rotation vs. Lateral Force
    axs[1, 1].plot(displacements, reactions[:, 0], label="Lateral Force X")
    axs[1, 1].plot(displacements, reactions[:, 1], label="Lateral Force Y")
    axs[1, 1].set_xlabel("Displacement [deg]")
    axs[1, 1].set_ylabel("Lateral Force [N]")
    axs[1, 1].set_title("Rotation vs. Lateral Force")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Rotation vs. Tipping Moment
    axs[1, 2].plot(displacements, reactions[:, 3], label="Tipping Moment X")
    axs[1, 2].plot(displacements, reactions[:, 4], label="Tipping Moment Y")
    axs[1, 2].set_xlabel("Displacement [deg]")
    axs[1, 2].set_ylabel("Tipping Moment [N·m]")
    axs[1, 2].set_title("Rotation vs. Tipping Moment")
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

    return results


reaction_plots = False
if reaction_plots:
    # Define the range of displacements to test
    disp_range_trans = 0.1 * radius  # Adjust this range as necessary for translation
    disp_range_rot = 4  # Degrees for rotation

    # Generate the plots
    results = plot_displacement_vs_reaction(PS, OFC, disp_range_trans, disp_range_rot)

# %%% Stability regions with recursive bisection


def binary_search_stability(
    theta: float, r_inner: float, r_outer: float, tolerance: float, logger
) -> Tuple[float, float]:
    low, high = r_inner, r_outer
    stable, unstable = low, high
    count = 0

    while high - low > tolerance:
        count += 1
        mid = (high + low) / 2
        x0 = mid * np.cos(np.radians(theta))
        y0 = mid * np.sin(np.radians(theta))

        logger.info(
            f"Theta: {theta:.2f}, Low: {low/length:.4f} [D], High: {high/length:.4f} [D], Mid: {mid/length:.4f} [D]"
        )
        stable_result, pos = trajectory_from_initial_conditions(x0=x0, y0=y0)
        if stable_result:
            stable = mid
            low = mid
            direction = "outwards"
        else:
            unstable = mid
            high = mid
            direction = "inwards"

        logger.info(
            f"Theta: {theta:.2f}, Mid: {mid/length :.4f} [D], Stable: {stable_result}, Direction: {direction}\n"
        )

    if stable is None:
        stable = low
    if unstable is None:
        unstable = high

    logger.info(
        f"Solution found after {count+2} iterations. Stable: {stable/length:.4f} [D], Unstable: {unstable/length:.4f} [D]"
    )
    return stable, unstable


def sweep_polar_coordinates(
    radius: float, d_theta: float, tolerance: float, logger, theta_max=90
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = time.time()
    angles = np.arange(0, theta_max + d_theta, d_theta)
    stable_radii = []
    unstable_radii = []
    positions = []

    for theta in angles:
        logger.info(f"\n\nStarting binary search for theta: {theta:.2f} degrees")

        if len(stable_radii) == 0:
            r_inner, r_outer = 0, radius * 0.5
        else:
            r_inner, r_outer = stable_radii[-1] / 1.2, unstable_radii[-1] * 1.2

        # Check if both inner and outer are stable or unstable
        inner_stable, _ = trajectory_from_initial_conditions(
            x0=r_inner * np.cos(np.radians(theta)),
            y0=r_inner * np.sin(np.radians(theta)),
        )
        outer_stable, _ = trajectory_from_initial_conditions(
            x0=r_outer * np.cos(np.radians(theta)),
            y0=r_outer * np.sin(np.radians(theta)),
        )

        if inner_stable == outer_stable:
            logger.info(
                f"Theta: {theta:.2f} degrees, pre-check caught: {r_inner/length=:.2f} [D],  {r_outer/length=:.2f} [D]"
            )
            r_inner, r_outer = 0.00, radius

        stable, unstable = binary_search_stability(
            theta, r_inner, r_outer, tolerance, logger
        )

        stable_radii.append(stable)
        unstable_radii.append(unstable)
        positions.append((theta, stable, unstable))
    end = time.time()
    dt = end - start
    logger.info(f"All in all that took {dt/60:.0f}m and {dt%60}s")
    return np.array(stable_radii), np.array(unstable_radii), np.array(positions)


def store_results(
    stable: np.ndarray, positions: np.ndarray, theta: np.ndarray, marking: int
) -> None:
    np.savetxt(
        f"temp/initial_pos_sweep/sweep_trajectories_stable_{marking}_sigma{sigma:.2g}_damping{damping_pressure:.2g}.csv",
        stable,
    )
    np.savetxt(
        f"temp/initial_pos_sweep/sweep_trajectories_positions_{marking}_sigma{sigma:.2g}_damping{damping_pressure:.2g}.csv",
        positions,
    )
    np.savetxt(
        f"temp/initial_pos_sweep/sweep_trajectories_theta_{marking}_sigma{sigma:.2g}_damping{damping_pressure:.2g}.csv",
        theta,
    )


def plot_results(stable: np.ndarray, theta: np.ndarray, radius: float) -> None:
    length = 2 * radius
    stable_interp = interp1d(theta, stable, bounds_error=False)

    # Define Cartesian grid
    max_radius = stable.max() * 1.25
    x_cartesian = np.linspace(-max_radius, max_radius, 500)
    y_cartesian = np.linspace(-max_radius, max_radius, 500)
    x_grid, y_grid = np.meshgrid(x_cartesian, y_cartesian)

    r_grid = np.sqrt(x_grid**2 + y_grid**2)
    theta_grid = np.rad2deg(np.arctan2(y_grid, x_grid)) % 360
    r_cutoff = stable_interp(
        theta_grid,
    )
    mask = r_grid < r_cutoff

    stable_matrix = np.zeros_like(x_grid)
    stable_matrix[mask] = 1

    # Plot the stability regions
    stability_contours = plt.figure(figsize=[8, 6])

    ax2 = stability_contours.add_subplot(111)
    stab_region = ax2.contourf(
        x_grid / length, y_grid / length, stable_matrix, cmap="binary"
    )
    plt.xlabel("x [D]")
    plt.ylabel("y [D]")
    ax2.set_aspect("equal")

    stable_patch = mpatches.Patch(color="black", label="stable region")
    unstable_patch = mpatches.Patch(color="grey", label="unstable")
    plt.legend(handles=[stable_patch, unstable_patch])

    stability_contours.tight_layout()
    plt.show()

    return None

    # Create a grid in polar coordinatesm
    max_radius = stable.max() * 1.25
    radii = set(stable)
    radii.add(0)
    radii.add(max_radius)
    radii = np.array(list(radii))
    radii.sort()
    theta_grid, radius_grid = np.meshgrid(theta, radii)

    # Initialize the stability matrix in polar coordinates
    stable_matrix = np.zeros_like(theta_grid)

    for i, theta_val in enumerate(theta):
        stable_radius = stable[i]
        for j in range(len(stable_matrix)):
            if radius_grid[j, i] <= stable_radius:
                stable_matrix[j, i] = 1  # Mark as stable

    # Convert polar coordinates to Cartesian coordinates
    x_polar = radius_grid * np.cos(np.radians(theta_grid))
    y_polar = radius_grid * np.sin(np.radians(theta_grid))

    # Define Cartesian grid
    x_cartesian = np.linspace(0, max_radius, 500)
    y_cartesian = np.linspace(0, max_radius, 500)
    x_grid, y_grid = np.meshgrid(x_cartesian, y_cartesian)

    # Interpolate the stability and altitude data onto the Cartesian grid
    stable_cartesian = griddata(
        (x_polar.flatten(), y_polar.flatten()),
        stable_matrix.flatten(),
        (x_grid, y_grid),
        method="nearest",
    )

    # Plot the stability regions
    stability_contours = plt.figure(figsize=[8, 6])

    ax2 = stability_contours.add_subplot(111)
    stab_region = ax2.contourf(
        x_grid / length, y_grid / length, stable_cartesian, cmap="binary", levels=1
    )
    cbar = stability_contours.colorbar(stab_region)
    cbar.set_label("Stability (1 = stable)")
    plt.xlabel("x [D]")
    plt.ylabel("y [D]")
    ax2.set_aspect("equal")

    stability_contours.tight_layout()
    plt.show()


recursive_trajectory_sweep = True
if recursive_trajectory_sweep:
    # PS setup
    PS.initial_positions, _ = PS.x_v_current

    # Setting up the logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture all messages
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set the logger level to INFO

    # Parameters for the sweep
    d_theta = 4  # Angular step in degrees
    tolerance = 0.01 * radius  # Tolerance for binary search
    theta_max = 360

    # Running the sweep
    marking = int(time.time())
    stable_radii, unstable_radii, positions = sweep_polar_coordinates(
        radius, d_theta, tolerance, logger, theta_max
    )

    # Storing the results
    store_results(
        stable_radii, positions, np.arange(0, theta_max + d_theta, d_theta), marking
    )

    # Preparing altitude data for plotting (assuming altitude is derived from positions in some way)
    altitude = np.array([pos[2] for pos in positions])

    # Plotting the results
    stable = (positions[:, 1] + positions[:, 2]) / 2
    theta = positions[:, 0]
    plot_results(stable, theta, radius)
