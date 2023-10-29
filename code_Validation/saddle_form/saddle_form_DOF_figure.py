"""
Script for PS framework performance testing, benchmark case where shape is sought of self-tensioned net
"""
import matplotlib.pyplot as plt
import numpy as np
# import saddle_form_input as input
from saddle_form_input import params as p
from saddle_form_input import connectivity_matrix, initial_conditions
import timeit
import time
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem


def converge(psystem: ParticleSystem, c: int):
    cond = False

    if c == 1:
        for m in range(p['t_steps']):           # viscous damping
            psystem.simulate()
            if np.linalg.norm(psystem.f_int) <= 1e-3:
                cond = True
                break

    elif c == 2:
        for m in range(p['t_steps']):  # kin damping without q
            psystem.kin_damp_sim()
            if np.linalg.norm(psystem.f_int) <= 1e-3:
                cond = True
                break

    if not cond:
        print("convergence not reached, option:", c)
        # print(psystem)
    return


if __name__ == "__main__":
    # result settings
    times = np.zeros((2, ))
    samplesize = 10
    n = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    grid_height = 5
    grid_length = 10
    k = [6e4]
    p['m'] = 1
    p['c'] = 1
    p['dt'] = 1
    p['t_steps'] = 1000
    p["l0"] = 0  # np.sqrt( 2 * (grid_length/(grid_size-1))**2)
    DOF = []
    t_vis = []
    t_kin = []
    for amount in n:
        print()
        grid_size = amount
        p['n'] = grid_size ** 2 + (grid_size - 1) ** 2

        # looping simulations
        time1 = time.time()
        for i in range(1, 3):
            for j in range(samplesize):
                print(i, j)
                # recalculate parameters

                c_m, f_nodes = connectivity_matrix(grid_size)
                # print(grid_size, p["m"], f_nodes, input.grid_height, input.grid_length)
                ic = initial_conditions(grid_size, p["m"], f_nodes, grid_height, grid_length)

                # instantiate PS with correct settings
                ps = ParticleSystem(c_m, ic, p)

                # measure convergence time
                times[i-1] += timeit.repeat(lambda: converge(ps, i), repeat=1, number=1)
        time2 = time.time()
        print(f'total loop time: {(time2 - time1):.4f} s')

        # printing statistics
        times = times/samplesize
        print(f"input settings: h = {p['dt']}, n = {p['n']}, k = {p['k']}, c = {p['c']}")
        print("average runtimes")
        print(f"PS vis damp: {times[0]:.3f}")
        print(f"PS kin damp: {times[1]:.3f}")

        DOF.append(p['n']*3)
        t_vis.append(times[0])
        t_kin.append(times[1])

    fig, ax = plt.subplots()

    plt.plot(DOF, t_vis)
    plt.plot(DOF, t_kin)

    ax.set_yscale('log')
    plt.xlabel('DOF [-]')
    plt.ylabel('convergence time [s]')
    plt.title('')
    plt.legend(['PS with viscous damping', 'PS with kinetic damping'])
    plt.grid()
    plt.show()
