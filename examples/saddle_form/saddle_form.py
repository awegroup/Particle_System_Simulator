"""
Script for PS framework validation, benchmark case where saddle form of self stressed network is sought
"""

import numpy as np
import saddle_form_input as input
import matplotlib.pyplot as plt
import pandas as pd
import time
from PSS.particleSystem import ParticleSystem


def print_simulation_settings():
    """Print all simulation parameters and settings"""
    print("=" * 60)
    print("SADDLE FORM SIMULATION SETTINGS")
    print("=" * 60)

    # Grid parameters
    print("\nGrid Parameters:")
    print(f"  Grid size: {input.grid_size}")
    print(f"  Grid length: {input.grid_length}")
    print(f"  Grid height: {input.grid_height}")
    print(f"  Total number of particles: {input.params['n']}")

    # Model parameters
    print("\nModel Parameters:")
    print(f"  Spring stiffness (k_t): {input.params['k_t']} N/m")
    print(f"  Segment stiffness (k): {input.params['k']} N/m")
    print(f"  Damping coefficient (c): {input.params['c']} N·s/m")
    print(f"  Tether length (L): {input.params['L']} m")
    print(f"  Block mass (m_block): {input.params['m_block']} kg")
    print(f"  Tether density (rho_tether): {input.params['rho_tether']} kg/m")
    print(f"  Segment mass (m_segment): {input.params['m_segment']} kg")
    print(f"  Rest length (l0): {input.params['l0']} m")

    # Simulation settings
    print("\nSimulation Settings:")
    print(f"  Time step (dt): {input.params['dt']} s")
    print(f"  Number of time steps: {input.params['t_steps']}")
    print(f"  Total simulation time: {input.params['dt'] * input.params['t_steps']} s")
    print(f"  Absolute tolerance: {input.params['abs_tol']}")
    print(f"  Relative tolerance: {input.params['rel_tol']}")
    print(f"  Maximum iterations: {int(input.params['max_iter'])}")

    # Physical parameters
    print("\nPhysical Parameters:")
    print(f"  Gravitational acceleration (g): {input.params['g']} m/s²")
    print(f"  Wind velocity (v_w): {input.params['v_w']} m/s")
    print(f"  Air density (rho): {input.params['rho']} kg/m³")
    print(f"  Drag coefficient bridle (c_d_bridle): {input.params['c_d_bridle']}")
    print(f"  Bridle diameter (d_bridle): {input.params['d_bridle']} m")

    # Connection information
    print(f"\nConnectivity:")
    print(f"  Total connections: {len(input.connections)}")

    # Fixed nodes information
    fixed_nodes = [i for i, cond in enumerate(input.init_cond) if cond[3]]
    print(f"  Fixed nodes: {len(fixed_nodes)} out of {len(input.init_cond)}")
    print(f"  Fixed node indices: {fixed_nodes}")

    print("=" * 60)
    print()


def instantiate_ps():
    return ParticleSystem(input.connections, input.init_cond, input.params)


def plot(psystem: ParticleSystem, psystem2: ParticleSystem):

    psystem.stress_self()
    psystem2.stress_self()

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

    position2 = pd.DataFrame(index=t_vector, columns=x)
    final_step2 = 0

    n = input.params["n"]
    f_ext = np.array([[0, 0, 0] for i in range(n)]).flatten()

    start_time = time.time()
    for (
        step
    ) in t_vector:  # propagating the simulation for each timestep and saving results
        position.loc[step], _ = psystem.simulate(f_ext)
        final_step = step
        if np.linalg.norm(psystem.f_int) <= 1e-3:
            print("Classic PS converged")
            break
    stop_time = time.time()

    start_time2 = time.time()
    for (
        step
    ) in t_vector:  # propagating the simulation for each timestep and saving results
        position2.loc[step], _ = psystem2.kin_damp_sim(f_ext)
        final_step2 = step
        if np.linalg.norm(psystem2.f_int) <= 1e-3:
            print("Kinetic damping PS converged")
            break
    stop_time2 = time.time()

    classic_runtime = stop_time - start_time
    kinetic_runtime = stop_time2 - start_time2

    print(f"PS classic: {classic_runtime:.4f} s")
    print(f"PS kinetic: {kinetic_runtime:.4f} s")

    # plotting & graph configuration
    # Data from layout after 1 iteration step
    X = []
    Y = []
    Z = []
    for i in range(n):
        X.append(position[f"x{i + 1}"].iloc[0])
        Y.append(position[f"y{i + 1}"].iloc[0])
        Z.append(position[f"z{i + 1}"].iloc[0])

    # Create color array based on fixed nodes
    colors = []
    for i in range(n):
        if input.init_cond[i][3]:  # Check if node is fixed (4th element is True)
            colors.append("blue")
        else:
            colors.append("red")

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # data from final timestep
    X_f = []
    Y_f = []
    Z_f = []
    for i in range(n):
        X_f.append(position[f"x{i + 1}"].loc[final_step])
        Y_f.append(position[f"y{i + 1}"].loc[final_step])
        Z_f.append(position[f"z{i + 1}"].loc[final_step])

    # plot inital layout
    ax.scatter(X, Y, Z, c=colors)
    for i, j, *_ in input.connections:
        ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]], color="black")

    # plot final found shape
    ax2.scatter(X_f, Y_f, Z_f, c=colors)
    for i, j, *_ in input.connections:
        ax2.plot([X_f[i], X_f[j]], [Y_f[i], Y_f[j]], [Z_f[i], Z_f[j]], color="black")

    ax2.text2D(
        0.05,
        0.99,
        f"Classic PS, runtime: {classic_runtime:.4f}s",
        transform=ax2.transAxes,
    )
    ax2.text2D(
        0.05,
        0.94,
        f"Kinetic PS, runtime: {kinetic_runtime:.4f}s",
        transform=ax2.transAxes,
    )

    # surf = ax.plot_surface(X, Y, Z)#, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    # plt.xlabel("time [s]")
    # plt.ylabel("position [m]")
    # plt.title("Validation PS framework, deflection of particles by wind flow, with Implicit Euler scheme")
    # plt.legend([f"displacement particle {i + 1}" for i in range(n)] + [f"kinetic damped particle {i + 1}" for i in range(n)])
    # plt.grid()

    # # saving resulting figure
    # figure = plt.gcf()
    # figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper
    #
    # # Not sure if this is the smartest way to automate saving results relative to other users directories
    # file_path = sys.path[1] + "/Msc_Alexander_Batchelor/code_Validation/benchmark_results/" \
    #                           "tether_deflection_windFlow/"
    # img_name = f"{input.params['n']}Particles-{input.params['k_t']}stiffness-{input.params['c']}damping_coefficient-" \
    #            f"{input.params['dt']}timestep-{input.params['t_steps']}steps.jpeg"
    # plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()
    return


if __name__ == "__main__":
    print_simulation_settings()

    ps = instantiate_ps()
    ps2 = instantiate_ps()

    plot(ps, ps2)
