"""
Input file for verification of correct implementation damping force, by modeling damped harmonic oscillator
"""
import numpy as np


def connectivity_matrix():
    matrix = [[0, 1]]
    return matrix


def initial_conditions():
    conditions = [[[0, 0, 0], [0, 0, 0], 1, True], [[0, 0, 1], [0, 0, 0], 1, False]]
    return conditions


def element_parameters(k, l0, c):
    e_p = [[k, l0, c]]
    return e_p


# dictionary of required parameters
params = {
    # model parameters
    "n": 2,                         # [-] number of particles
    "k": 1000,                       # [N/m] spring coefficient
    "c": 2*1000**0.5,               # [N s/m] critical damping coefficient (2 * sqrt(k/m))
    # "c": 10,                        # [N s/m] damping coefficient
    "L": 0,                         # [m] tether length

    # simulation settings
    "dt": 0.1,                    # [s] simulation timestep
    "t_steps": 25,                # [-] number of simulated time steps
    "abs_tol": 1e-50,               # [m/s] absolute error tolerance iterative solver
    "rel_tol": 1e-5,                # [-] relative error tolerance iterative solver
    "max_iter": 1e5,                # [-] maximum number of iterations

    # physical parameters
    "g": 0                          # [m/s^2] EXCLUDED gravitational acceleration
}
# calculated parameters
params["l0"] = params["L"]/(params["n"]-1)

# instantiate connectivity matrix and initial conditions array
c_matrix = connectivity_matrix()
init_cond = initial_conditions()
elem_param = element_parameters(params["k"], params["l0"], params["c"])
