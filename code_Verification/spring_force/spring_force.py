"""
Script for verification of correct implementation spring force of SpringDamper object within ParticleSystem framework
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
import sys
from Particle_System_Simulator.particleSystem.ParticleSystem import ParticleSystem


def connectivity_matrix():
    matrix = [[0, 1]]
    return matrix


def initial_conditions():
    conditions = [[[0, 0, 0], [0, 0, 0], 1, True], [[0, 0, 1], [0, 0, 0], 1, False]]
    return conditions


def element_parameters(k, l0, c):
    e_m = [[k, l0, c]]
    return e_m


def instantiate_ps():

    # dictionary of required parameters
    params = {
        # model parameters
        "n": 2,  # [-] number of particles
        "k": 2e4,  # [N/m] spring stiffness
        "c": 0,  # [N s/m] damping coefficient
        "L": 0,  # [m] tether length
        # simulation settings
        "dt": 0.001,  # [s] simulation timestep
        "t_steps": 1000,  # [-] number of simulated time steps
        "abs_tol": 1e-50,  # [m/s] absolute error tolerance iterative solver
        "rel_tol": 1e-5,  # [-] relative error tolerance iterative solver
        "max_iter": 1e5,  # [-] maximum number of iterations
        # physical parameters
        "g": 9.81,  # [m/s^2] gravitational acceleration
    }
    # calculated parameters
    params["l0"] = params["L"] / (params["n"] - 1)

    # instantiate connectivity matrix and initial conditions array
    c_matrix = connectivity_matrix()
    init_cond = initial_conditions()
    elem_params = element_parameters(params["k"], params["l0"], params["c"])

    # instantiate ParticleSystem object
    return ParticleSystem(c_matrix, init_cond, params)


def exact_solution(t_vector: npt.ArrayLike):  # analytical solution for this test case
    k = input.params["k"]
    m = input.init_cond[1][-2]
    omega = np.sqrt(k / m)

    exact_x = np.cos(omega * t_vector)

    # Estimated (expected) decay rate of implicit Euler scheme as a function of t
    dt = input.params["dt"]
    decay = np.exp(-0.5 * omega**2 * dt * t_vector)

    # # Estimated (expected) phase shift (freq. reduction) of implicit Euler scheme as a function of t
    # import cmath
    # real = np.zeros((len(t_vector,)))
    # phase_shift = np.exp(-omega * t_vector * (1 - 1/3 * (omega * dt)**2))
    # imaginary_vector = list(zip(real, phase_shift))
    # shift = cmath.phase(complex(imaginary_vector[-1][0], imaginary_vector[-1][1]))
    # print(f"Estimated phase shift: {shift:.3f} pi")
    # NOT ACCURATE, NEEDS MORE THINKING TROUGH

    return exact_x, decay


def plot(psystem: ParticleSystem):  # visualization of simulation and analytical results

    # time vector for simulation loop, data storage and plotting
    t_vector = np.linspace(
        0, input.params["t_steps"] * input.params["dt"], input.params["t_steps"] + 1
    )

    # DataFrames as storage method of choice
    x = {
        "x": np.zeros(
            len(t_vector),
        )
    }
    v = {
        "v": np.zeros(
            len(t_vector),
        )
    }
    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)

    # addition of (constant) external forces
    f_ext = np.zeros(
        input.params["n"] * 3,
    )

    for (
        step
    ) in t_vector:  # propagating the simulation for each timestep and saving results
        x_next, v_next = psystem.simulate(f_ext)
        # x_next, v_next = psystem.kin_damp_sim(f_ext)
        position.loc[step], velocity.loc[step] = x_next[-1], v_next[-1]

    # generating analytical solution for the same time vector
    exact, decay = exact_solution(t_vector)

    # correcting simulation for decay rate
    corrected = np.divide(np.array(position["x"]), decay)

    # plot without correction
    position.plot()
    plt.plot(t_vector, exact, "g")
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Simulation of internal spring force, without damping or external loads")
    plt.legend(
        [
            f"PS simulation, dt = {input.params['dt']} s, k = {input.params['k']:.1f}",
            "Exact solution",
        ]
    )
    plt.grid()
    plt.show()

    # plot with correction
    position.plot()
    plt.plot(t_vector, exact, "g")
    plt.plot(t_vector, decay, "r")
    plt.plot(t_vector, corrected, "k--", lw=2)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Verification PS spring force implementation with Implicit Euler scheme")
    plt.legend(
        [
            f"PS simulation, dt = {input.params['dt']} s, k = {input.params['k']:.1f}",
            "Exact solution",
            "Decay rate implicit Euler",
            "corrected for decay rate",
        ]
    )
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    # Not sure if this is the smartest way to automate saving results relative to other users directories
    file_path = sys.path[1] + "/code_Verification/spring_force/results/"
    img_name = f"{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['dt']}timestep.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches="tight")

    plt.show()

    return


if __name__ == "__main__":

    ps = instantiate_ps()

    plot(ps)
