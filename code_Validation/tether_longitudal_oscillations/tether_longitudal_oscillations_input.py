"""
Input file for validation of PS, benchmark case of longitudal tether oscillations due to dropped mass
"""
import numpy as np


def connectivity_matrix(n: int):
    # matrix = np.eye(n, k=1) + np.eye(n, k=-1)
    matrix = []
    for i in range(n-1):
        matrix.append([i, i+1])
    return matrix


def initial_conditions(l0: float, n: int, m_segment: float, m_block: float):
    conditions = [[[0, 0, (n-1)*l0-i*l0], [0, 0, 0], m_segment, False] for i in range(n)]
    conditions[0][-1] = True       # Set top end of tether to fixed
    conditions[0][-2] -= 0.5 * m_segment
    conditions[-1][-2] += m_block - 0.5 * m_segment
    return conditions


def element_parameters(k, l0, c, n):
    e_m = []
    for i in range(n-1):
        e_m.append([k, l0, c])
    return e_m


# dictionary of required parameters
params = {
    # model parameters
    "n": 5,                         # [-] number of particles
    "k": 119575.9,                       # [N/m] spring stiffness
    "c": 92,                         # [N s/m] damping coefficient
    "L": 5.14,                        # [m] tether length
    "m_block": 327.8,                  # [kg] mass attached to end of tether
    "rho_tether": 0.012,              # [kg/m] mass density tether

    # simulation settings
    "dt": 0.01,                    # [s] simulation timestep
    "t_steps": 1000,                # [-] number of simulated time steps
    "abs_tol": 1e-5,                # [m/s] absolute error tolerance iterative solver
    "rel_tol": 1e-4,                # [-] relative error tolerance iterative solver
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
init_cond = initial_conditions(params["l0"], params["n"], params["m_segment"], params["m_block"])


m = [init_cond[i][-2] for i in range(params['n'])]
params["c"] = 2 * np.sqrt(params['k'] / np.sum(m))

elem_params = element_parameters(params["k"], params["l0"], params["c"], params["n"])

# print(init_cond, params["k"])
# print([params["k"] / (i + 1) for i in range(params["n"] - 1)])
# m = [init_cond[i][-2] for i in range(params['n'])]
# print(m)
# print([sum(m[i + 1:]) for i in range(params["n"] - 1)])
