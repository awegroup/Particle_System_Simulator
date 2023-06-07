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

from Msc_Alexander_Batchelor.src.AnalysisModules.SystemEnergy import system_energy


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


def s_energy(pos, vel, t_vector):
    k = input.params['k']
    c = input.params['c']
    dt = t_vector[1]
    m = input.init_cond[1][-2]

    # Elastic potential
    ep = 0.5 * k * pos**2

    # # Gravitational potential
    # gp = m * g * h                # gravity not included in this test case

    # Kinetic energy
    ke = 0.5 * m * vel**2

    # Energy dissipated by friction
    ed = pd.DataFrame(index=t_vector, columns={'ed': np.zeros(len(t_vector))})
    ed.iloc[0] = 0
    for i in range(1, len(t_vector)):
        ed.iloc[i] = c * dt * abs(vel.iloc[i]**2 - vel.iloc[i - 1]**2)

    # Total system energy
    total_energy = ep['x'] + ke['v'] - ed['ed']

    # print("ep:", ep)
    # print("ke:", ke)
    # print("ed:", ed)
    # print("te:", total_energy)

    return total_energy


def plot(psystem: ParticleSystem):          # visualization of simulation and analytical results

    # time vector for simulation loop, data storage and plotting
    dt = input.params['dt']
    t_steps = input.params["t_steps"]
    t_vector = np.linspace(0,  t_steps * dt, t_steps)

    # DataFrames as storage method of choice
    x = {"x": np.zeros(len(t_vector), )}
    v = {"v": np.zeros(len(t_vector), )}
    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)
    sys_en = pd.DataFrame(index=t_vector, columns={'SE': np.zeros(len(t_vector))})
    v_prev = np.zeros(input.params['n'] * 3)

    # addition of (constant) external forces
    f_ext = np.zeros(input.params['n'] * 3, )

    for step in t_vector:          # propagating the simulation for each timestep and saving results
        sys_en.loc[step] = system_energy(psystem, input.params, v_prev)
        x_next, v_next = psystem.simulate(f_ext)
        position.loc[step], velocity.loc[step] = x_next[-1], v_next[-1]
        v_prev = v_next
    # print(sys_en)

    # calculating system energy over time
    energy = s_energy(position, velocity, t_vector)
    norm_energy = energy / energy.iloc[0]
    sys_en = sys_en / sys_en.iloc[0]

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
               f"{input.params['dt']}timestep-{t_vector[-1]:.1f}s.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    # separate plot for system energy
    plt.figure()
    fig = norm_energy.plot()
    sys_en.plot(ax=fig)
    plt.xlabel("time [s]")
    plt.ylabel("energy [j]")
    plt.title("System energy for test case damping force, normalized for initial system energy")
    plt.legend(["Normalized system energy over time", "new module"])
    plt.grid()
    plt.show()

    plt.show()
    return


if __name__ == "__main__":
    ps = instantiate_ps()

    plot(ps)
