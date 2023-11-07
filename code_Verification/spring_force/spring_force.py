"""
Script for verification of correct implementation spring force of SpringDamper object within ParticleSystem framework
"""
import numpy as np
import input_spring_force as input
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
import sys
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.elem_params, input.params)


def exact_solution(t_vector: npt.ArrayLike):                       # analytical solution for this test case
    k = input.params["k"]
    m = input.init_cond[1][-2]
    omega = np.sqrt(k/m)

    exact_x = np.cos(omega * t_vector)

    # Estimated (expected) decay rate of implicit Euler scheme as a function of t
    dt = input.params['dt']
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


def plot(psystem: ParticleSystem):          # visualization of simulation and analytical results

    # time vector for simulation loop, data storage and plotting
    t_vector = np.linspace(0, input.params["t_steps"] * input.params["dt"], input.params["t_steps"] + 1)

    # DataFrames as storage method of choice
    x = {"x": np.zeros(len(t_vector),)}
    v = {"v": np.zeros(len(t_vector), )}
    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)

    # addition of (constant) external forces
    f_ext = np.zeros(input.params['n'] * 3, )

    for step in t_vector:          # propagating the simulation for each timestep and saving results
        x_next, v_next = psystem.simulate(f_ext)
        # x_next, v_next = psystem.kin_damp_sim(f_ext)
        position.loc[step], velocity.loc[step] = x_next[-1], v_next[-1]

    # generating analytical solution for the same time vector
    exact, decay = exact_solution(t_vector)

    # correcting simulation for decay rate
    corrected = np.divide(np.array(position["x"]), decay)

    # plot without correction
    position.plot()
    plt.plot(t_vector, exact, 'g')
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Simulation of internal spring force, without damping or external loads")
    plt.legend([f"PS simulation, dt = {input.params['dt']} s, k = {input.params['k']:.1f}", "Exact solution"])
    plt.grid()
    plt.show()

    # plot with correction
    position.plot()
    plt.plot(t_vector, exact, 'g')
    plt.plot(t_vector, decay, 'r')
    plt.plot(t_vector, corrected, 'k--', lw=2)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Verification PS spring force implementation with Implicit Euler scheme")
    plt.legend([f"PS simulation, dt = {input.params['dt']} s, k = {input.params['k']:.1f}",
                "Exact solution", "Decay rate implicit Euler", "corrected for decay rate"])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    # Not sure if this is the smartest way to automate saving results relative to other users directories
    file_path = sys.path[1] + "/Msc_Alexander_Batchelor/code_Verification/verification_results/spring_force/"
    img_name = f"{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['dt']}timestep.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()

    return


if __name__ == "__main__":

    ps = instantiate_ps()

    plot(ps)
