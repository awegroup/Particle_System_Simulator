"""
Script for verification of correct implementation external gravitational force within ParticleSystem framework
"""
import numpy as np
import numpy.typing as npt
import input_gravitational_force as input
import matplotlib.pyplot as plt
import pandas as pd
import sys
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.params)


def external_forces(n: int, g: float):
    return


def exact_solution(t_vector: npt.ArrayLike):
    g = input.params["g"]
    exact_x = -0.5 * g * t_vector**2
    return exact_x


def plot(psystem: ParticleSystem):

    # time vector for simulation loop, data storage and plotting
    t_vector = np.linspace(0, input.params["t_steps"] * input.params["dt"], input.params["t_steps"] + 1)

    # DataFrames as storage method of choice
    x = {"x": np.zeros(len(t_vector),)}
    v = {"v": np.zeros(len(t_vector), )}
    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)

    # addition of (constant) external forces
    f_ext = np.array([[0, 0, -input.params['g']] for i in range(input.params['n'])]).flatten()

    for step in t_vector:
        x_next, v_next = psystem.simulate(f_ext)
        position.loc[step], velocity.loc[step] = x_next[-1], v_next[-1]

    exact = exact_solution(t_vector)

    # graph configuration
    position.plot()
    plt.plot(t_vector, exact)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Verification external gravitational force")
    plt.legend(["PS simulation", "Exact solution"])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    # Not sure if this is the smartest way to automate saving results relative to other users directories
    file_path = sys.path[1] + "/Msc_Alexander_Batchelor/code_Verification/verification_results/gravitational_force/"
    img_name = f"{input.params['n']}Particles-{input.params['dt']}timestep.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()

    return


if __name__ == "__main__":
    ps = instantiate_ps()

    plot(ps)
