"""
Script for PS framework validation, benchmark case where saddle form of self stressed network is sought
"""
import numpy as np
import hencky_problem_input as input
import matplotlib.pyplot as plt
import pandas as pd
import time
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.element_param, input.params)


def force(ps: ParticleSystem):
    p_t = input.p                   # transverse uniform pressure
    e_l = input.element_list        # list containing particles forming quadrilateral elements
    particles = ps.particles

    vectors = []
    for element in e_l:
        p1 = particles[element[0]].x
        p2 = particles[element[1]].x
        p3 = particles[element[2]].x
        p4 = particles[element[3]].x

        v1 = p2 - p1
        v2 = p3 - p2
        normal_vector = np.cross(v1, v2)
        normal_vector = normal_vector/np.linalg.norm(normal_vector)
        x_avg = (p1[0] + p2[0] + p3[0] + p4[0]) / 4.0
        y_avg = (p1[1] + p2[1] + p3[1] + p4[1]) / 4.0
        z_avg = (p1[2] + p2[2] + p3[2] + p4[2]) / 4.0

        vectors.append([x_avg, y_avg, z_avg, normal_vector])

    return vectors


def calc_f(ps: ParticleSystem):
    p_t = input.p                   # transverse uniform pressure
    e_l = input.element_list        # list containing particles forming quadrilateral elements
    particles = ps.particles
    force_vector = np.zeros(len(particles)*3, )

    for element in e_l:
        p1 = particles[element[0]].x
        p2 = particles[element[1]].x
        p3 = particles[element[2]].x

        v1 = p2 - p1
        v2 = p3 - p2
        normal_vector = np.cross(v1, v2)
        normal_vector = normal_vector/np.linalg.norm(normal_vector)
        area = np.linalg.norm(normal_vector) / 2.0

        force_value = area*p_t
        for corner_particle_index in element:
            force_vector[corner_particle_index*3:(corner_particle_index+1)*3] += 0.25*force_value*normal_vector

    return force_vector

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
    f_ext = calc_f(ps)

    start_time = time.time()
    for step in t_vector:           # propagating the simulation for each timestep and saving results
        f_ext = calc_f(ps)
        position.loc[step], _ = psystem.simulate(f_ext)
        for i, particle in enumerate(ps.particles):      # need to exclude fixed particles for force-based convergence
            if particle.fixed:
                f_ext[i*3:(i+1)*3] = 0
        final_step = step
        if np.linalg.norm(psystem.f_int+f_ext) <= 1e-3:
            print("Classic PS converged")
            break
    stop_time = time.time()

    start_time2 = time.time()
    for step in t_vector:  # propagating the simulation for each timestep and saving results
        f_ext = calc_f(psystem2)
        position2.loc[step], _ = psystem2.kin_damp_sim(f_ext)
        for i, particle in enumerate(psystem2.particles):
            if particle.fixed:
                f_ext[i*3:(i+1)*3] = 0
        final_step2 = step
        if np.linalg.norm(psystem2.f_int+f_ext) <= 1e-3:
            print("Kinetic damping PS converged")
            break
    stop_time2 = time.time()

    print(f'PS classic: {(stop_time - start_time):.4f} s')
    print(f'PS kinetic: {(stop_time2 - start_time2):.4f} s')

    # plotting & graph configuration
    # Data final step PS viscous damping
    X = []
    Y = []
    Z = []
    for i in range(n):
        X.append(position[f"x{i + 1}"].loc[final_step])
        Y.append(position[f"y{i + 1}"].loc[final_step])
        Z.append(position[f"z{i + 1}"].loc[final_step])

    fig= plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Data final step PS kinetic damping
    X_f = []
    Y_f = []
    Z_f = []
    for i in range(n):
        X_f.append(position2[f"x{i + 1}"].loc[final_step2])
        Y_f.append(position2[f"y{i + 1}"].loc[final_step2])
        Z_f.append(position2[f"z{i + 1}"].loc[final_step2])

    # plot inital layout
    ax.scatter(X, Y, Z, c='red')
    for indices in input.c_matrix:
        ax.plot([X[indices[0]], X[indices[1]]], [Y[indices[0]], Y[indices[1]]], [Z[indices[0]], Z[indices[1]]],
                color='black')

    # plot final found shape
    ax2.scatter(X_f, Y_f, Z_f, c='red')
    for indices in input.c_matrix:
        ax2.plot([X_f[indices[0]], X_f[indices[1]]], [Y_f[indices[0]], Y_f[indices[1]]], [Z_f[indices[0]],
                Z_f[indices[1]]], color='black')

    # surf = ax.plot_surface(X, Y, Z)#, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    # plt.xlabel("time [s]")
    # plt.ylabel("position [m]")

    ax.set_title("PS viscous damping found Hencky")
    ax2.set_title("PS kinetic damping found shape")
    plt.show()
    return


if __name__ == "__main__":
    ps = instantiate_ps()
    ps2 = instantiate_ps()

    plot(ps, ps2)

    # test to check if normal vectors for force calculation are pointed in right direction
    # for i in range(10):
    #     f_ext = calc_f(ps)
    #     ps.simulate(f_ext)
    #     v = force(ps)
    #     x = []
    #     y = []
    #     z = []
    #     for particle in ps.particles:
    #         x.append(particle.x[0])
    #         y.append(particle.x[1])
    #         z.append(particle.x[2])
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection="3d")
    #
    #     ax.scatter(x, y, z, c='red')
    #     for indices in input.c_matrix:
    #         ax.plot([x[indices[0]], x[indices[1]]], [y[indices[0]], y[indices[1]]], [z[indices[0]], z[indices[1]]],
    #                 color='black')
    #     for vector in v:
    #         x = vector[0]
    #         y = vector[1]
    #         z = vector[2]
    #         sf = 0.03
    #         x_u = vector[3][0] * sf
    #         y_u = vector[3][1] * sf
    #         z_u = vector[3][2] * sf
    #         ax.quiver(x, y, z, x_u, y_u, z_u, color='r')
    #     plt.show()
