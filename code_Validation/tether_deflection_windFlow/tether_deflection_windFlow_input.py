"""
Input file for validation of PS, benchmark case of tether, fixed at both ends, deflected by perpendicular wind flow
"""
import numpy as np


def connectivity_matrix(n: int):
    matrix = np.eye(n, k=1) + np.eye(n, k=-1)
    return matrix


def initial_conditions(l0: float, n: int, m_segment: float):
    conditions = [[[0, 0, (n-1)*l0-i*l0], [0, 0, 0], m_segment, False] for i in range(n)]
    conditions[0][-1] = conditions[-1][-1] = True       # Set ends of tether to fixed
    return conditions


# dictionary of required parameters
params = {
    # model parameters
    "n": 5,                         # [-]       number of particles
    "k_t": 2933.3,                # [N/m]     spring stiffness
    "c": 1,                         # [N s/m] damping coefficient
    "L": 300,                        # [m]       tether length
    "m_block": 100,                  # [kg]     mass attached to end of tether
    "rho_tether": 0.012,              # [kg/m]    mass density tether

    # simulation settings
    "dt": 1,                    # [s]       simulation timestep
    "t_steps": 10000,                # [-]      number of simulated time steps
    "abs_tol": 1e-5,               # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,                # [-]       relative error tolerance iterative solver
    "max_iter": 1e5,                # [-]       maximum number of iterations

    # physical parameters
    "g": 9.807,                     # [m/s^2]   gravitational acceleration
    "v_w": [6, 0, 0],               # [m/s]     wind velocity vector
    'rho': 1.225,                   # [kg/ m3]  air density
    'c_d_bridle': 1.05,             # [-]       drag-coefficient of bridles
    "d_bridle": 0.02                # [m]       diameter of bridle lines
}
# calculated parameters
params["l0"] = params["L"] / (params["n"] - 1)
params["m_segment"] = params["L"] * params["rho_tether"] / (params["n"] - 1)
params["k"] = params["k_t"] * (params["n"] - 1)           # segment stiffness

# instantiate connectivity matrix and initial conditions array
c_matrix = connectivity_matrix(params['n'])
init_cond = initial_conditions(params["l0"], params["n"], params["m_segment"])

