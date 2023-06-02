"""
Script for verification of correct implementation spring force of SpringDamper object within ParticleSystem framework
"""
import numpy as np
import numpy.typing as npt
import input_damping_force as input
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

import sys
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.params)


def exact_solution(t_vector: npt.ArrayLike):
    k = input.params["k"]
    c = input.params["c"]
    m = input.init_cond[1][-2]
    omega = np.sqrt(k / m)
    gamma = c/m
    zeta = c/(2 * omega)        # critical damping faction

    # Analytical solution depends on value of zeta
    if zeta <1:
        print("system is underdamped")
    elif zeta == 1:
        print("system is critically damped")
    else:
        print("system is overdamped")

    # solve as IVP
    y0 = np.array([1, 0])

    def syst_of_diff_eq(t, y):
        A = np.array([[0, 1], [-omega**2, -gamma]])
        system = np.matmul(A, y)
        return system

    t_int = [t_vector[0], t_vector[-1]]
    ivp_solution = solve_ivp(syst_of_diff_eq, t_int, y0=y0, t_eval=t_vector)

    # # exact solution
    # c1 = 0.5
    # c2 = 0.5
    # exact_x = np.exp(-.5 * c * t_vector) * (c1 * np.exp(t_vector * np.sqrt((c/2)**2 - omega**2)
    #                                                     + c2 * np.exp(-t_vector * np.sqrt((c/2)**2 - omega**2))))

    # Estimated (expected) decay rate of implicit Euler scheme as a function of t
    dt = input.params['dt']
    decay = np.exp(-0.5 * omega ** 2 * dt * t_vector)

    return ivp_solution, decay


def plot(psystem: ParticleSystem):          # visualization of simulation and analytical results

    # time vector for simulation loop, data storage and plotting
    t_vector = np.linspace(input.params["dt"], input.params["t_steps"] * input.params["dt"], input.params["t_steps"] + 1)

    # DataFrames as storage method of choice
    x = {"x": np.zeros(len(t_vector), )}
    v = {"v": np.zeros(len(t_vector), )}
    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)

    # addition of (constant) external forces
    f_ext = np.zeros(input.params['n'] * 3, )

    for step in t_vector:          # propagating the simulation for each timestep and saving results
        x_next, v_next = psystem.simulate(f_ext)
        position.loc[step], velocity.loc[step] = x_next[-1], v_next[-1]

    # generating analytical solution for the same time vector
    exact, decay = exact_solution(t_vector)

    # correcting simulation for decay rate
    corrected = np.divide(np.array(position["x"]), decay)

    # graph configuration
    position.plot()
    plt.plot(t_vector, exact.y[0])
    plt.plot(t_vector, decay)
    plt.plot(t_vector, corrected)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Verification PS damping force implementation with Implicit Euler scheme")
    plt.legend(["PS simulation", "Exact solution", "Decay rate implicit Euler", "corrected for decay rate"])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    # Not sure if this is the smartest way to automate saving results relative to other users directories
    file_path = sys.path[1] + "/Msc_Alexander_Batchelor/code_Verification/verification_results/damping_force/"
    img_name = f"{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['c']:.3f}dampingC-" \
               f"{input.params['dt']}timestep.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()

    return


if __name__ == "__main__":
    ps = instantiate_ps()

    plot(ps)
