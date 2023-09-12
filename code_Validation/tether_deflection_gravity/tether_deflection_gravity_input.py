"""
Input file for validation of PS, benchmark case of tether, fixed at both ends, deflected by perpendicular gravity
"""
import numpy as np


def connectivity_matrix(n: int):
    matrix = np.eye(n, k=1) + np.eye(n, k=-1)
    return matrix


def initial_conditions(l0: float, n: int, m_segment: float):
    conditions = [[[l0*i, 0, 0], [0, 0, 0], m_segment, False] for i in range(n)]
    conditions[0][-1] = conditions[-1][-1] = True       # Set ends of tether to fixed
    return conditions


# dictionary of required parameters
params = {
    # model parameters
    "n": 5,                         # [-] number of particles
    "k": 2e3,                       # [N/m] spring stiffness
    "c": 100,                         # [N s/m] damping coefficient
    "L": 10,                        # [m] tether length
    "m_block": 100,                  # [kg] mass attached to end of tether
    "rho_tether": 0.1,              # [kg/m] mass density tether

    # simulation settings
    "dt": 0.01,                    # [s] simulation timestep
    "t_steps": 1000,                # [-] number of simulated time steps
    "abs_tol": 1e-50,               # [m/s] absolute error tolerance iterative solver
    "rel_tol": 1e-5,                # [-] relative error tolerance iterative solver
    "max_iter": 1e5,                # [-] maximum number of iterations

    # physical parameters
    "g": 9.807                      # [m/s^2] gravitational acceleration
}
# calculated parameters
params["l0"] = params["L"] / (params["n"] - 1)
params["m_segment"] = params["L"] * params["rho_tether"] / (params["n"] - 1)
params["k"] = params["k"] * (params["n"] - 1)           # segment stiffness

# instantiate connectivity matrix and initial conditions array
c_matrix = connectivity_matrix(params['n'])
init_cond = initial_conditions(params["l0"], params["n"], params["m_segment"])
