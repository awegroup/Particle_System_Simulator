"""
Script for PS framework validation, benchmark case where tether is fixed at both ends and is deflected by perpendicular
gravity which results in a catenary line
"""

import numpy as np
import numpy.typing as npt
import tether_deflection_gravity_input as input
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
from Particle_System_Simulator.particleSystem import ParticleSystem

from sympy import *


def instantiate_ps():
    return ParticleSystem(
        input.c_matrix, input.init_cond, input.elem_params, input.params
    )


def generate_animation(pos, n: int, t: npt.ArrayLike):
    from matplotlib import animation
    import matplotlib
    import math

    matplotlib.rcParams["animation.ffmpeg_path"] = r"C:\\FFmpeg\\bin\\ffmpeg.exe"
    filename = (
        f"Gravity_deflection-{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['c']}"
        f"damping_coefficient-{input.params['dt']}timestep-{input.params['t_steps']}steps-.mov"
    )
    savelocation = r"C:\\Users\\Alexander\\Documents\\Master\\Thesis\\Figures\\GIFs\\"

    # configuration of plot
    fig, ax = plt.subplots()
    ax.set_xlim((-1, 11))
    ax.set_ylim((-1, 0.5))
    ax.grid(which="major")
    plt.ylabel("height [m]")
    plt.xlabel("x position [m]")
    plt.title(f"Animation of tether deflected by gravity")

    # calculation which values for each frame
    fps = 60  # 1 / input.params['dt']
    multi = round(input.params["dt"] ** -1 / fps)
    n_frames = math.floor(len(t) / multi)
    frame_indeces = [i * multi for i in range(n_frames)]

    (line,) = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        index = frame_indeces[i]
        timestep = t[index]
        x = pos.loc[timestep, [f"x{j + 1}" for j in range(n)]]
        y = pos.loc[timestep, [f"z{j + 1}" for j in range(n)]]
        line.set_data(x, y)
        return (line,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=20, blit=True
    )  # , save_count=len(self.t))

    writervideo = animation.FFMpegWriter(fps=fps)
    anim.save(savelocation + filename, writer=writervideo)
    plt.cla()
    return


def analytical_solution(sag: float, t_l: float):
    # catenary line equation analytical solution
    L = input.params["L"]
    h = abs(sag)
    a = (0.25 * t_l**2 - h**2) / (2 * h)
    x = np.linspace(0, L, 1000)
    y = a * np.cosh(
        (x - 0.5 * L) / a
    )  # shift curve 0.5*L in direction of positive x-axis
    y -= y[0]  # adjust height
    return x, y


def plot(psystem: ParticleSystem):
    n = input.params["n"]
    t_vector = np.linspace(
        input.params["dt"],
        input.params["t_steps"] * input.params["dt"],
        input.params["t_steps"],
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

    g = input.params["g"]
    n = input.params["n"]
    # f_ext = np.array([[0, 0, -g] for i in range(n)]).flatten()
    particles = ps.particles
    f_ext = np.array([[0, 0, -g * particle.m] for particle in particles]).flatten()
    f_check = f_ext.copy()
    f_check[0:3] = f_check[-2:] = 0

    start_time = time.time()
    for (
        step
    ) in t_vector:  # propagating the simulation for each timestep and saving results
        # position.loc[step], velocity.loc[step] = psystem.simulate(f_ext)
        # position.loc[step], velocity.loc[step] = psystem.kin_damp_sim(f_ext)
        position.loc[step], velocity.loc[step] = psystem.kin_damp_sim(
            f_ext, q_correction=True
        )

        residual_f = f_check + psystem.f_int
        # print(np.linalg.norm(residual_f))
        # print(input.params)
        if np.linalg.norm(residual_f) <= 1e-3:
            print("convergence criteria satisfied")
            break
    stop_time = time.time()
    print(f"convergence time: {(stop_time - start_time):.4f} s")

    # generate animation of results, requires smarter configuration to make usable on other PCs
    # generate_animation(position, n, t_vector)

    # generating analytical solution for the same time vector
    tether_length = 0
    particles = ps.particles
    for i in range(n - 1):
        tether_length += np.linalg.norm(particles[i].x - particles[i + 1].x)
    x_pos = position["x2"].count() - 1
    x, y = analytical_solution(min(position.iloc[x_pos]), tether_length)

    # plotting & graph configuration
    plt.figure(0)
    for i in range(1, n - 1):
        position[f"z{i + 1}"].plot()
    # plt.plot(t, exact)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Particle deflection of PS with kinetic damping without q-correction")
    plt.legend([f"displacement particle {i + 1}" for i in range(1, n - 1)])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    # Not sure if this is the smartest way to automate saving results relative to other users directories
    file_path = (
        sys.path[1] + "/Msc_Alexander_Batchelor/code_Validation/benchmark_results/"
        "tether_deflection_gravity/"
    )
    img_name = (
        f"{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['c']}damping_coefficient-"
        f"{input.params['dt']}timestep-{input.params['t_steps']}steps.jpeg"
    )
    plt.savefig(file_path + img_name, dpi=300, bbox_inches="tight")

    # plot of end state simulation vs analytical solution
    plt.figure(1)
    simx = []
    simy = []
    for i in range(n):
        simx.append(position[f"x{i + 1}"].iloc[x_pos])
        simy.append(position[f"z{i + 1}"].iloc[x_pos])

    plt.plot(simx, simy, lw=4)
    plt.plot(x, y)
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    # plt.title(f"Analytical caternary and resulting catenary of PS with viscous damping, n = {input.params['n']}")
    plt.title(
        f"Analytical caternary and resulting catenary of PS with kinetic damping with q correction, n = {input.params['n']}"
    )
    plt.grid()
    plt.legend(["Simulation final state particles", "Analytical catenary"])
    plt.show()

    return


if __name__ == "__main__":
    ps = instantiate_ps()

    plot(ps)
