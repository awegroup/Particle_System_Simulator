# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:26:14 2023

@author: Mark
"""

"""
Script for PS framework validation, benchmark case where saddle form of self stressed network is sought
"""
import numpy as np
import saddle_form_input as input
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import pandas as pd
import time
from PSS.particleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.connections, input.init_cond, input.params)


def animate(i, n, psystem, position, b, lines, scatters):
    plt.ion()
    X, Y, Z = [], [], []
    for j in range(n):
        X.append(position[f"x{j+1}"].iloc[i])
        Y.append(position[f"y{j+1}"].iloc[i])
        Z.append(position[f"z{j+1}"].iloc[i])

    scatters._offsets3d = (X, Y, Z)

    for k, indices in enumerate(b):
        lines[k].set_data(
            [X[indices[0]], X[indices[1]]], [Y[indices[0]], Y[indices[1]]]
        )
        lines[k].set_3d_properties([Z[indices[0]], Z[indices[1]]])
    return (scatters, lines)


def plot(psystem: ParticleSystem, psystem2: ParticleSystem):
    n = input.params["n"]
    t_vector = np.linspace(
        input.params["dt"],
        input.params["t_steps"] * input.params["dt"],
        input.params["t_steps"],
    )

    x = {}
    for i in range(n):
        x[f"x{i + 1}"] = np.zeros(len(t_vector))
        x[f"y{i + 1}"] = np.zeros(len(t_vector))
        x[f"z{i + 1}"] = np.zeros(len(t_vector))

    position = pd.DataFrame(index=t_vector, columns=x)
    final_step = 0

    f_ext = np.array([[0, 0, 0] for i in range(n)]).flatten()

    start_time = time.time()
    for (
        step
    ) in t_vector:  # propagating the simulation for each timestep and saving results
        position.loc[step], _ = psystem2.kin_damp_sim(f_ext)
        final_step = step
        if np.linalg.norm(psystem2.f_int) <= 1e-3:
            print("Kinetic damping PS converged")
            break
    stop_time = time.time()

    print(f"PS kinetic: {(stop_time - start_time):.4f} s")

    frames = t_vector[: np.where(t_vector == final_step)[0][0] + 1]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    b = np.nonzero(np.triu(input.connections))
    b = np.column_stack((b[0], b[1]))

    X, Y, Z = [], [], []
    for j in range(n):
        X.append(position[f"x{j+1}"].iloc[0])
        Y.append(position[f"y{j+1}"].iloc[0])
        Z.append(position[f"z{j+1}"].iloc[0])

    scatters = ax.scatter(X, Y, Z, c="red")

    lines = []
    for indices in b:
        (line,) = ax.plot(
            [X[indices[0]], X[indices[1]]],
            [Y[indices[0]], Y[indices[1]]],
            [Z[indices[0]], Z[indices[1]]],
            color="black",
        )
        lines.append(line)

    ani = FuncAnimation(
        fig,
        animate,
        frames=frames,
        fargs=(n, psystem, position, b, lines, scatters),
        interval=100,
        blit=False,
    )
    plt.show()

    return position, b, t_vector, final_step


if __name__ == "__main__":
    ps = instantiate_ps()
    ps2 = instantiate_ps()

    position, b, t_vector, final_step = plot(ps, ps2)
