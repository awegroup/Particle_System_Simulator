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
import time
from PSS.particleSystem import ParticleSystem
from PSS.AnalysisModules import system_energy


def instantiate_ps():
    return ParticleSystem(
        input.c_matrix, input.init_cond, input.elem_params, input.params
    )


def exact_solution(t_vector: npt.ArrayLike):
    # analytical steady state solution for particles position
    k = input.params["k"]
    c = input.params["c"]
    n = input.params["n"]
    g = input.params["g"]
    m = [input.init_cond[i][-2] for i in range(n)]

    omega = np.sqrt(k / np.sum(m))

    # construct DataFrame with length of t_vector to store steady state particle displacement (x" = 0, x' = 0)
    x = {f"steady_state_p{i + 1}": np.zeros(len(t_vector)) for i in range(n - 1)}
    exact_x = pd.DataFrame(index=t_vector, columns=x)

    steady_state_displacement = np.array(
        [np.sum(m[i + 1 :]) * -g / (k / (i + 1)) for i in range(n - 1)]
    )

    for step in t_vector:
        exact_x.loc[step] = steady_state_displacement

    # Estimated (expected) decay rate of implicit Euler scheme as a function of t
    dt = input.params["dt"]
    decay = np.exp(-0.5 * omega**2 * dt * t_vector)

    zeta = c / (2 * omega)  # critical damping faction

    # (Estimated) system damping, based on
    if zeta < 1:
        print("system is underdamped")
    elif zeta == 1:
        print("system is critically damped")
    else:
        print("system is overdamped")

    return exact_x, decay


def plot(psystem: ParticleSystem, psystem2: ParticleSystem, psystem3: ParticleSystem):
    n = input.params["n"]
    t_vector = np.linspace(
        0, input.params["t_steps"] * input.params["dt"], input.params["t_steps"] + 1
    )

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
    sys_energy = pd.DataFrame(index=t_vector, columns={"E": np.zeros(len(t_vector))})
    v_prev = np.zeros(
        n * 3,
    )

    g = input.params["g"]
    n = input.params["n"]

    m = [input.init_cond[i][-2] for i in range(n)]
    f_ext = np.array([[0, 0, -g * m[i]] for i in range(n)]).flatten()
    f_check = f_ext.copy()
    f_check[:3] = 0
    m = np.array(
        [[input.init_cond[i][-2] for j in range(3)] for i in range(n)]
    ).flatten()
    m = np.diag(m)

    start_time = time.time()
    for (
        step
    ) in t_vector:  # propagating the simulation for each timestep and saving results
        if step == 0:
            x, v = psystem.x_v_current
            position.loc[step], velocity.loc[step] = x, v
            sys_energy.loc[step] = np.matmul(np.matmul(v_prev, m), v_prev)
            f_1 = f_ext.copy()
            v_1 = -g * np.sqrt(0.1 / (0.5 * g))
            f_1[-3:] = [0, 0, 10 * v_1 / input.params["dt"]]
            x_next, v_next = psystem.simulate(f_1)
            v_prev = v_next
            continue
        # sys_energy.loc[step] = system_energy(psystem, input.params, v_prev)
        sys_energy.loc[step] = np.matmul(np.matmul(v_prev, m), v_prev)

        x_next, v_next = psystem.simulate(f_ext)
        # x_next, v_next = psystem.kin_damp_sim(f_ext)
        # x_next, v_next = psystem.kin_damp_sim(f_ext, q_correction=True)

        position.loc[step], velocity.loc[step] = x_next, v_next
        v_prev = v_next

        residual_f = f_check + psystem.f_int
        if np.linalg.norm(residual_f) <= 1e-3:
            print("convergence criteria satisfied")
            break
    stop_time = time.time()
    print(f"convergence time: {(stop_time - start_time):.4f} s")

    # calculating system energy over time
    norm_energy = sys_energy  # / sys_energy.iloc[0]

    # generating analytical solution for the same time vector
    exact, decay = exact_solution(t_vector)

    # plotting & graph configuration
    for i in range(1, n):
        position[f"z{i + 1}"] -= input.init_cond[i][0][-1]
        position[f"z{i + 1}"].plot()

    x_length = position["z2"].count()
    plt.plot(t_vector[:x_length], exact.iloc[:x_length], ls="--")

    for i in range(
        n - 1
    ):  # setting particle colors equivalent to their analytical solution
        color = plt.gca().lines[i].get_color()
        plt.gca().lines[i + n - 1].set_color(color)

    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    # plt.title("Benchmark 1, PS with viscous damping simulation of longitudal tether oscillations")
    plt.title("Benchmark 1, PS with kinetic damping simulation with q correction")
    plt.legend(
        [f"displacement particle {i + 2}" for i in range(n - 1)]
        + [f"Analytical steady state pos. particle {i + 2}" for i in range(n - 1)]
    )
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    # Not sure if this is the smartest way to automate saving results relative to other users directories
    file_path = (
        sys.path[1] + "/Msc_Alexander_Batchelor/code_Validation/benchmark_results/"
        "tether_longitudal_oscillations/"
    )
    img_name = (
        f"{input.params['n']}Particles-{input.params['k']:.3f}stiffness-{input.params['c']:.3f}damping_coefficient-"
        f"{input.params['dt']}timestep-{input.params['t_steps']}steps.jpeg"
    )
    plt.savefig(file_path + img_name, dpi=300, bbox_inches="tight")

    # separate plot for system energy
    norm_energy.plot()
    plt.xlabel("time [s]")
    plt.ylabel("energy [J]")
    plt.title("Kinetic energy of longitudal oscillations benchmark system")
    plt.legend(["kinetic energy"])
    plt.grid()

    # error between analytical and steady-state positions
    for i in range(1, n):
        print(
            f"error p{i}: ",
            exact[f"steady_state_p{i}"].iloc[x_length - 1]
            - position[f"z{i+1}"].iloc[x_length - 1],
        )

    # frequency analysis

    fourier = np.fft.fft(position[f"z{n}"].iloc[: x_length - 1])
    plt.figure(3)

    T = input.params["dt"]
    N = x_length
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    yf = 2.0 / N * np.abs(fourier[: N // 2])
    print("main frequency peak [hz]:", xf[np.where(yf == max(yf[2:]))])
    plt.plot(xf, 2.0 / N * np.abs(fourier[: N // 2]))
    plt.xlabel("frequency [hz]")
    plt.ylabel("|y(f)|")
    plt.title("fft of benchmark 1")
    plt.grid()
    plt.show()

    return


if __name__ == "__main__":
    ps = instantiate_ps()
    ps2 = instantiate_ps()
    ps3 = instantiate_ps()

    plot(ps, ps2, ps3)
