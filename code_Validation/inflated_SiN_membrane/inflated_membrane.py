# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:10:12 2023

@author: Mark Kalsbeek


Validation case based on data from:
"Poisson's Ratio of Low-Temperature PECVD Silicon Nitride Thin Films" Walmsley et al, 2007
http://ieeexplore.ieee.org/document/4276823/

In short: They created square and rectangular membranes from low and high stress SiN. Then inflated them using gass pressure and reported measured deflections.

Available data:

Test data:
- crosssectional bulge profiles through the center of a square membrane and a rectangular membrane under 300 Pa of pressure for a 305-nm-thick PECVD SiNxHy film that was deposited at 205 ◦C
- Measured relationship between applied pressure and bulge height for square and rectangular PECVD SiNxHy membranes that were deposited at (a) 125 ◦C and (b) 205 ◦C

Curve-fitting parameters for the pressure-bulge height values:

| SiN:H_y Deposition Temp. (°C) | A1/s (kPa/µm) | A2/s (kPa/µm^3) | A1/r (kPa/µm) | A2/r (kPa/µm^3) |
|-------------------------------|---------------|-----------------|---------------|-----------------|
| 125                           | 1.094x10^-2   | 1.630x10^-4     | 6.462x10^-3   | 9.490x10^-5     |
| 205                           | 3.426x10^-2   | 1.259x10^-4     | 2.014x10^-3   | 7.291x10^-5     |

Resulting material properties:
| SiN:H_y Deposition Temperature (°C) | E (GPa)   | σ0 (MPa)  |
|-------------------------------------|-----------|-----------|
| 125                                 | 79 ± 4    | 5.1 ± 1.2 |
| 205                                 | 151 ± 6   | 29 ± 3    |

Geometrical information:
    The materials from different temperatures also differ in thickness:
        680 and 305 nm for the 125 ◦C and 205 ◦C deposited films, respectively.


Note:
    There is an inconsistency in the test data.
    The bulge in the square membrane case measures 6.8 um in inkscape.
    Plugging that into the provided fitted curve I find a reported pressure of 273 Pa.
    While the figure claims that this was for a pressure of 300 Pa.
    This is a 9.0% discrepancy.
    The other way around, plugging in 300 Pa I find an expected bulge-height of 7.32 um.
    This is a 7.6% discrepancy.

Initial stress:
    Due to differential coefficients of thermal expansion the SiN has internal stresses
    To r epresent them we will pre-stretch the mesh using the stress_self command
    Because it is a bi-axial stress state we will use strain = sigma_0 / E + v sigma_0 / E
    Where v is Poissons' ratio
"""
import logging

import numpy as np
import matplotlib.pyplot as plt

from src.particleSystem.ParticleSystem import ParticleSystem
from src.Sim.simulations import Simulate_airbag
import src.Mesh.mesh_functions as MF

curve_fitting_parameters = {
    "125": {
        "A1/s": 1.094e-02,
        "A2/s": 1.630e-04,
        "A1/r": 6.462e-03,
        "A2/r": 9.490e-05
    },
    "205": {
        "A1/s": 3.426e-02,
        "A2/s": 1.259e-04,
        "A1/r": 2.014e-02,
        "A2/r": 7.291e-05
    }
}

curve_fitting_parameters_analytical = {
    "125": {
        "A1/s": 1.304e-02,
        "A2/s": 1.582e-04
    },
    "205": {
        "A1/s": 3.325e-02,
        "A2/s": 1.383e-04
    }
}

material_properties = {
    "125": {
        "E [GPa]": 79, # +/- 4
        "Sigma_0 [MPa]": 5.1 # +/- 1.5
    },
    "205": {
        "E [GPa]": 151, # +/- 6
        "Sigma_0 [MPa]": 29 # +/- 3
    },
    'rho': 3.17e3 # [kg/m3]
}

# Misread the paper maybe? there are two sets and which is the right one is unclear
material_properties_alternative = {
    "125": {
        "E [GPa]": 82,
        "Sigma_0 [MPa]": 4.3
    },
    "205": {
        "E [GPa]": 146,
        "Sigma_0 [MPa]": 32
    },
    'rho': 3.17e3 # [kg/m3]
}

params = {
    # model parameters
    "k": 250,  # [N/m]   spring stiffness
    "k_d": 250,  # [N/m] spring stiffness for diagonal elements
    "c": 10,  # [N s/m] damping coefficient
    "m_segment": 1, # [kg] mass of each node

    # simulation settings
    "dt": 1e-5,  # [s]       simulation timestep
#    'adaptive_timestepping':0, # [m] max distance traversed per timestep
    "t_steps": 1e4,  # [-]      number of simulated time steps
    "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-10,  # [-]       relative error tolerance iterative solver
    "max_iter": 1e2,  # [-]       maximum number of iterations]
    "convergence_threshold": 1e-25, # [-]
    "min_iterations": 10, # [-]

    # sim specific settigns
    "pressure": 3e2, # [Pa] inflation pressure

    # testcase specific parameters
    # Geometric parameters
    "length": 1900 * 1e-6, # [m]
    "width": 1900 * 1e-6, # [m]
    "thickness": 305e-9, # [m]

    # designations
    "temperature": '205', # used as dict key to acces material properties
    "fitting_paramater_1":"A1/s", # These are used to get the right input data
    "fitting_paramater_2":"A2/s"
    }

params_square_high_t= params.copy()
params_square_high_t['testcase_name'] = "Square Membrane, T=205"
params_square_high_t['ratio_k_k_d']= 0.3  # Coefficient determined for this mesh shape to provide poissons ratio of 0.25

params_rectangle_high_t = params.copy()
params_rectangle_high_t['testcase_name'] = 'Rectangluar Membrane, T=205'
params_rectangle_high_t['ratio_k_k_d']= 0.3 # Coefficient determined for this mesh shape to provide poissons ratio of 0.25
params_rectangle_high_t['fitting_paramater_1'] = "A1/r"
params_rectangle_high_t['fitting_paramater_2'] = "A2/r"
params_rectangle_high_t['length'] = 9000e-6 # [m]
params_rectangle_high_t['convergence_threshold'] *=9000/1900 # Adjust convergence crit for larger system (more mass = more E_kin)

params_square_low_t = params_square_high_t.copy()
params_square_low_t['testcase_name'] = 'Square Membrane, T=125'
params_square_low_t['ratio_k_k_d']= 0.2755  # Coefficient determined for this mesh shape to provide poissons ratio of 0.23
params_square_low_t['temperature'] = '125'
params_square_low_t['thickness'] = 680e-9

params_rectangle_low_t = params_rectangle_high_t.copy()
params_rectangle_low_t['testcase_name'] = 'Rectangluar Membrane, T=125'
params_rectangle_low_t['ratio_k_k_d']= 0.2755 # Coefficient determined for this mesh shape to provide poissons ratio of 0.25
params_rectangle_low_t['temperature'] = '125'
params_rectangle_low_t['thickness'] = 680e-9

testcases = [params_square_high_t, params_rectangle_high_t, params_square_low_t, params_rectangle_low_t]

def setup_testcase(params=params, n_segments = 20):
    diagonal_spring_ratio = 0.306228 # Coefficient determined for this mesh shape to provide poissons ratio of 0.25


    params['k'] = material_properties[params['temperature']]["E [GPa]"]*1e9*params['thickness']*n_segments / (n_segments+1+2*n_segments*diagonal_spring_ratio/np.sqrt(2))
    #params['k'] /=31.47/29
    params['k_d'] = params['k']*diagonal_spring_ratio


    params['edge_length']= params['width']/n_segments
    params['adaptive_timestepping'] = params['edge_length']/100

    # Initialize the particle system
    initial_conditions, connections = MF.mesh_airbag_square_cross(params['length'],
                                                                  params['edge_length'],
                                                                  params = params  ,
                                                                  width = params['width'],
                                                                  noncompressive = True,
                                                                  sparse=False)

    PS = ParticleSystem(connections, initial_conditions,params)

    # Now we do some fine-tuning on the particle system
    # Assign the correct particle masses
    PS.calculate_correct_masses(params['thickness'],material_properties['rho'])

    # Calculate biaxial stiffness coefficient and apply pre_stress
    E = material_properties[params['temperature']]["E [GPa]"]*1e9
    strain_0 = (material_properties[params['temperature']]["Sigma_0 [MPa]"]*1e6)/E
    strain_0 /= 31.47/29 # Emperical factor to adjust pre-stress
    PS.stress_self(1/(1+strain_0))

    # Checking if pre-stess is correct
    f = PS._ParticleSystem__one_d_force_vector()
    f = f.reshape((len(f)//3,3))
    # Due to stress concentrations at edges will only look at a centered segment
    # There is still a question of what the correct way of looking at stress is here
    # Is it the stress felt in the center, or the average stress in the whole edge?
    # Do we see the stress concentrations at the edges as meaningfull or not?
    f_center = f[int(n_segments/2)]
    f_edge = f[::n_segments+1]
    f_edge_inner = f_edge[1:-1]

    A_cross = params['thickness'] * params['edge_length']
    sigma_0_center = f_center[1]/A_cross
    sigma_0_mean = np.sum(f_edge_inner, axis=0)[0]/(A_cross*(n_segments-0.5))
    print(f'Applied pre-stress is {sigma_0_center/1e6:.2f} [MPa] in the center, {sigma_0_mean/1e6:.2f} on average.')

    # Altering boundary conditions
    for particle in PS.particles:
        if particle.fixed:
            particle.set_fixed(True)
        if not particle.fixed: # adding some noise to help get it startd
            particle.x[2] += (np.random.random()-0.5) * 5e-7

    Sim = Simulate_airbag(PS, params)
    return Sim

#%% Sweep Pressure
def sweep_pressure(PS, ax = None):
    params = PS.params
    temp = params['temperature']
    height_bulge = np.poly1d(
        [curve_fitting_parameters[temp][params["fitting_paramater_2"]],
         0,
         curve_fitting_parameters[temp][params["fitting_paramater_1"]],
         0])

    pressures = np.linspace(0.1, 0.9, 17)

    z_sim_hist = []
    z_exp_hist = []
    print('='*60)
    print(f'Starting Simulation: {params["testcase_name"]}')
    Sim = Simulate_airbag(PS, params)
    for pressure in pressures:
        print('='*60)
        params['pressure'] = pressure*1e3
        Sim.run_simulation(plotframes=0,
                           plot_whole_bag = False,
                           printframes=0,
                           simulation_function = 'kinetic_damping',
                           both_sides=False)
        x, v = PS.x_v_current_3D
        z_max = x[:,2].max()
        z_sim_hist.append(z_max)

        z_exp = (height_bulge-pressure).roots.real
        z_exp = z_exp[z_exp>0]
        z_exp_hist.append(z_exp)

        print('-'*60)
        print(f'Pressure: \t\t\t {pressure:.2} [kPa]')
        print(f'Expected height:\t{z_exp[0]*1e-6:.2e} [m]')
        print(f'Simulated height:\t{z_max:.2e} [m]')
        print(f'Error: \t\t\t\t{(z_max*1e6-z_exp[0])/z_exp[0]*100:.1f}%')
        print('='*60)
        print('\n')
    z_sim_hist = np.array(z_sim_hist) * 1e6

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.plot(pressures, z_sim_hist, label=f'{params["testcase_name"]}: Simulated')
    ax.plot(pressures, z_exp_hist, label=f'{params["testcase_name"]}: Expected')
    ax.set_title('Simulated verus expected behaviour')
    ax.set_xlabel('pressure [kPa]')
    ax.set_ylabel('Bulge Height [$\mu m$]')
    ax.legend()
    return [ax, pressures, z_sim_hist, z_exp_hist]

#%% __main__ block
if __name__ == '__main__':
    debug = False
    if debug:
        Logger = logging.getLogger()
        Logger.setLevel(00) # default is Logger.setLevel(40)

    single= True
    if single:
        case = testcases[0]
        Sim = setup_testcase(case, n_segments=20)
        PS = Sim.PS
        PS.plot(colors='strain')

        Sim.run_simulation(plotframes=1,
                           plot_whole_bag = False,
                           printframes=1,
                           simulation_function = 'kinetic_damping',
                           both_sides=False)
        PS.plot(colors='strain')
        x, v = PS.x_v_current_3D
        v_mag = np.linalg.norm(v, axis =1)
        z_max = x[:,2].max()
        f_int = [sd.force_value() for sd in PS.springdampers]
        f_int_mag = np.linalg.norm(f_int, axis=1)


        temp = params['temperature']
        height_bulge = np.poly1d(
            [curve_fitting_parameters[temp][params["fitting_paramater_2"]],
             0,
             curve_fitting_parameters[temp][params["fitting_paramater_1"]],
             0])
        z_exp = (height_bulge-0.3).roots.real
        z_exp = z_exp[z_exp>0]
        print(f'Expected height:\t{z_exp[0]*1e-6:.2e} [m]')
        print(f'Simulated height:\t{z_max:.2e} [m]')
        print(f'Error: \t\t\t\t{(z_max*1e6-z_exp[0])/z_exp[0]*100:.1f}%')
    else:
        history = []
        fig = plt.figure()

        for i, case in enumerate(testcases):
            ax = plt.subplot(221+i)
            ax.margins(0,0)
            Sim = setup_testcase(case, n_segments=20)
            PS = Sim.PS
            result = sweep_pressure(PS, ax)
            history.append(result)

            x, _ = PS.x_v_current_3D
            z = x[:,2]
            zlim =  np.max([z.max(), 1e-7])/10

            fig3d = plt.figure()
            ax3d = fig3d.add_subplot(projection='3d')
            PS.plot(ax3d)#, colors='strain')
            ax3d.set_title(f"{case['testcase_name']}")
            ax3d.set_zlim(-zlim,zlim)
            fig3d.tight_layout()
            fig3d.savefig(f"temp\{case['testcase_name']}.jpg", dpi = 200, format = 'jpg')
            plt.close(fig3d)

        dump = []
        dump.append(history[0][1])
        for case in history:
            dump.append(case[2])
            dump.append(np.array(case[3]).T[0])

        import pandas as pd
        df = pd.DataFrame(dump).T

        cols = ['Pressure [kPa]', 'ws_sim,T=205','ws_exp,T=205','wr_sim,T=205','wr_exp,T=205', 'ws_sim,T=125','ws_exp,T=125','wr_sim,T=125','wr_exp,T=125']

        df.columns = cols
