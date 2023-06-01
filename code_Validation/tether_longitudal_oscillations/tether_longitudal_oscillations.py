"""
Script for PS framework validation, benchmark case where tether is fixed at top end and exhibits longitudal oscillations
due to a dropped mass fixed at its other end at t = 0.
"""
import numpy as np
import numpy.typing as npt
import tether_longitudal_oscillations_input as input
import matplotlib.pyplot as plt
import pandas as pd
import sys
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.params)


def exact_solution(t_vector: npt.ArrayLike):
    # analytical steady state solution for particles position
    k = input.params["k"]
    c = input.params["c"]
    n = input.params["n"]
    m = [input.init_cond[i][-2] for i in range(n)]

    omega = np.sqrt(k / m)
    dx = 0
    exact_x = [np.ones(len(t_vector)) * dx[i] for i in range(n)]

    # Estimated (expected) decay rate of implicit Euler scheme as a function of t
    dt = input.params['dt']
    decay = np.exp(-0.5 * omega ** 2 * dt * t_vector)

    zeta = c/(2 * omega)        # critical damping faction

    # Analytical solution depends on value of zeta
    if zeta <1:
        print("system is underdamped")
    elif zeta == 1:
        print("system is critically damped")
    else:
        print("system is overdamped")

    return exact_x, decay


def plot(psystem: ParticleSystem):
    n = input.params['n']
    t_vector = np.linspace(input.params["dt"], input.params["t_steps"] * input.params["dt"], input.params["t_steps"])

    x = {}
    v = {}
    for i in range(n):
        x[f"x{i + 1}"] = np.zeros(len(t_vector))
        x[f"y{i + 1}"] = np.zeros(len(t_vector))
        x[f"z{i + 1}"] = np.zeros(len(t_vector))
        v[f"vx{i + 1}"] = np.zeros(len(t_vector))
        v[f"vy{i + 1}"] = np.zeros(len(t_vector))
        v[f"vz{i + 1}"] = np.zeros(len(t_vector))

    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)

    g = input.params["g"]
    n = input.params["n"]
    f_ext = np.array([[0, 0, -g] for i in range(n)]).flatten()

    for step in t_vector:           # propagating the simulation for each timestep and saving results
        position.loc[step], velocity.loc[step] = psystem.simulate(f_ext)
    print(position)
    # generating analytical solution for the same time vector
    # exact, decay = exact_solution(t_vector)

    # plotting & graph configuration
    for i in range(n):
        position[f"z{i + 1}"] -= input.init_cond[i][0][-1]
        position[f"z{i + 1}"].plot()
    # plt.plot(t, exact)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Validation PS framework, longitudal oscillations of particles with Implicit Euler scheme")
    plt.legend([f"displacement particle {i + 1}" for i in range(n)])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    # Not sure if this is the smartest way to automate saving results relative to other users directories
    file_path = sys.path[1] + "/Msc_Alexander_Batchelor/code_Validation/benchmark_results/" \
                              "tether_longitudal_oscillations/"
    img_name = f"{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['c']}damping_coefficient-" \
               f"{input.params['dt']}timestep-{input.params['t_steps']}steps.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()

    return


if __name__ == "__main__":
    ps = instantiate_ps()

    plot(ps)
