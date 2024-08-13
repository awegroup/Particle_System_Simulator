"""
Script for verification of correct implementation external gravitational force within ParticleSystem framework
"""

import numpy as np
import numpy.typing as npt
import input_gravitational_force as input
import matplotlib.pyplot as plt
import pandas as pd
import sys
from Particle_System_Simulator.particleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(
        input.c_matrix, input.init_cond, input.elem_params, input.params
    )


def exact_solution(t_vector: npt.ArrayLike):
    g = input.params["g"]
    exact_x = -0.5 * g * t_vector**2
    return exact_x


def plot(psystem: ParticleSystem, ps2: ParticleSystem, ps3: ParticleSystem):

    # time vectors for simulation loop, data storage and plotting
    t_vector = np.linspace(
        input.params["dt"],
        input.params["t_steps"] * input.params["dt"],
        input.params["t_steps"],
    )
    t2 = np.linspace(0.1, 100 * 0.1, 100)
    t3 = np.linspace(1, 10 * 1, 10)

    # DataFrames as storage method of choice
    x = {
        "x": np.zeros(
            len(t_vector),
        )
    }
    position = pd.DataFrame(index=t_vector, columns=x)

    # addition of (constant) external forces
    f_ext = np.array(
        [[0, 0, -input.params["g"]] for i in range(input.params["n"])]
    ).flatten()

    for step in t_vector:
        x_next, _ = psystem.simulate(f_ext)
        position.loc[step] = x_next[-1]

    x2 = {
        "x": np.zeros(
            len(t2),
        )
    }
    p2 = pd.DataFrame(index=t2, columns=x2)

    for step in t2:
        x_next, _ = ps2.simulate(f_ext)
        p2.loc[step] = x_next[-1]

    x3 = {
        "x": np.zeros(
            len(t3),
        )
    }
    p3 = pd.DataFrame(index=t3, columns=x3)

    for step in t3:
        x_next, _ = ps3.simulate(f_ext)
        p3.loc[step] = x_next[-1]

    exact = exact_solution(t_vector)

    # graph configuration
    plt.plot(position, "k--", lw=2)
    plt.plot(p2)
    plt.plot(p3)
    plt.plot(t_vector, exact)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Simulation of constant external load, gravity, without internal forces")
    plt.legend(
        [
            f"PS simulation, dt = 0.01 s",
            f"PS simulation, dt = {input.params['dt']} s",
            f"PS simulation, dt = 1 s",
            "Exact solution",
        ]
    )
    plt.grid()
    #
    # # saving resulting figure
    # figure = plt.gcf()
    # figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper
    #
    # # Not sure if this is the smartest way to automate saving results relative to other users directories
    # file_path = sys.path[1] + "/Msc_Alexander_Batchelor/code_Verification/verification_results/gravitational_force/"
    # img_name = f"{input.params['n']}Particles-{input.params['dt']}timestep.jpeg"
    # plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()

    # plot for absolute error of simulation with timestep of 1s
    t_vector = [round(elem, 2) for elem in t_vector]
    for step in t3:
        i = t_vector.index(step)
        p3.loc[step] -= exact[i]

    plt.plot(abs(p3))
    plt.xlabel("time [s]")
    plt.ylabel("absolute error [m]")
    plt.title("Error growth over time for simulation with timestep of 1 s")
    plt.grid()
    plt.show()

    # plot showing local error propotional to timestep squared, property of first-order scheme
    h = np.linspace(0, 1.2, 13)
    r3 = abs(p3["x"].iloc[0])
    r1 = abs(position["x"].iloc[0] - exact[t_vector.index(0.01)]) / r3
    r2 = abs(p2["x"].iloc[0] - exact[t_vector.index(0.1)]) / r3
    r3 = r3 / r3

    print(r1, r2, r3)

    plt.plot(0.01, r1, "k", marker=".", markersize=10)
    plt.plot(0.1, r2, "b", marker=".", markersize=10)
    plt.plot(1, r3, "tab:orange", marker=".", markersize=10)

    plt.plot(h, h**2)
    plt.xlabel("time step value [s]")
    plt.ylabel("normalized error [-]")
    plt.title(
        "Normalized errors at the first simulation step and timestep squared curve"
    )
    plt.legend(["timestep 1 s", "timestep 0.1 s", "timestep 0.01 s", "quadratic of h"])
    plt.grid()
    plt.show()

    return


if __name__ == "__main__":
    ps = ParticleSystem(
        input.c_matrix, input.init_cond, input.elem_params, input.params
    )

    params = input.params.copy()
    params2 = input.params.copy()

    params["dt"] = 0.1
    params["t_steps"] = 100
    ps2 = ParticleSystem(input.c_matrix, input.init_cond, input.elem_params, params)

    params2["dt"] = 1
    params2["t_steps"] = 10
    ps3 = ParticleSystem(input.c_matrix, input.init_cond, input.elem_params, params2)

    plot(ps, ps2, ps3)
