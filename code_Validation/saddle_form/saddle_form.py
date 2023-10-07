"""
Script for PS framework validation, benchmark case where saddle form of self stressed network is sought
"""
import numpy as np
import saddle_form_input as input
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import time
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.params)


def plot(psystem: ParticleSystem, psystem2: ParticleSystem):
    n = input.params['n']
    t_vector = np.linspace(input.params["dt"], input.params["t_steps"] * input.params["dt"], input.params["t_steps"])

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
    for step in t_vector:           # propagating the simulation for each timestep and saving results
        position.loc[step], _ = psystem.simulate(f_ext)
        final_step = step
        if np.linalg.norm(psystem.f_int) <= 1e-3:
            print("Classic PS converged")
            break
    stop_time = time.time()

    start_time2 = time.time()
    for step in t_vector:  # propagating the simulation for each timestep and saving results
        position2.loc[step], _ = psystem2.kin_damp_sim(f_ext)
        final_step2 = step
        if np.linalg.norm(psystem2.f_int) <= 1e-3:
            print("Kinetic damping PS converged")
            break
    stop_time2 = time.time()

    print(f'PS classic: {(stop_time - start_time):.4f} s')
    print(f'PS kinetic: {(stop_time2 - start_time2):.4f} s')

    # plotting & graph configuration
    # Data from layout after 1 iteration step
    X = []
    Y = []
    Z = []
    for i in range(n):
        X.append(position[f"x{i + 1}"].iloc[0])
        Y.append(position[f"y{i + 1}"].iloc[0])
        Z.append(position[f"z{i + 1}"].iloc[0])

    fig= plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    b = np.nonzero(np.triu(input.c_matrix))
    b = np.column_stack((b[0], b[1]))

    # data from final timestep
    X_f = []
    Y_f = []
    Z_f = []
    for i in range(n):
        X_f.append(position[f"x{i + 1}"].loc[final_step])
        Y_f.append(position[f"y{i + 1}"].loc[final_step])
        Z_f.append(position[f"z{i + 1}"].loc[final_step])

    # plot inital layout
    ax.scatter(X, Y, Z, c='red')
    for indices in b:
        ax.plot([X[indices[0]], X[indices[1]]], [Y[indices[0]], Y[indices[1]]], [Z[indices[0]], Z[indices[1]]],
                color='black')

    # plot final found shape
    ax2.scatter(X_f, Y_f, Z_f, c='red')
    for indices in b:
        ax2.plot([X_f[indices[0]], X_f[indices[1]]], [Y_f[indices[0]], Y_f[indices[1]]], [Z_f[indices[0]],
                Z_f[indices[1]]], color='black')

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
    ps = instantiate_ps()
    ps2 = instantiate_ps()

    plot(ps, ps2)
