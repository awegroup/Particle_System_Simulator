"""
Script for PS framework performance testing, benchmark case where shape is sought of self-tensioned net
"""
import matplotlib.pyplot as plt
import numpy as np
# import saddle_form_input as input
from saddle_form_input import params as p
from saddle_form_input import connectivity_matrix, initial_conditions, element_parameters
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

    elif c == 3:
        for m in range(p['t_steps']):  # kin damping with q
            x_step, _ = psystem.kin_damp_sim(q_correction=True)
            if np.linalg.norm(psystem.f_int) <= 1e-3:
                cond = True
                break

    if not cond:
        print("convergence not reached, option:", c)
        # print(psystem)
    return


if __name__ == "__main__":
    # result settings
    times = np.zeros((3, ))
    samplesize = 10
    n = [3, 5, 8, 10]#, 20]
    # n = [20]
    k = [1, 100, 6e4, 1e6]

    grid_height = 5
    grid_length = 10
    p['m'] = 1
    p['c'] = 1
    p['dt'] = 1
    p['t_steps'] = 1000
    p["l0"] = 0  # np.sqrt( 2 * (grid_length/(grid_size-1))**2)

    for stiffness in k:
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
                    # p["k"] = p["k_t"] * (p["n"] - 1)  # segment stiffness
                    p['k'] = stiffness

                    c_m, f_nodes = connectivity_matrix(grid_size)
                    # print(grid_size, p["m"], f_nodes, input.grid_height, input.grid_length)
                    ic = initial_conditions(grid_size, p["m"], f_nodes, grid_height, grid_length)
                    e_m = element_parameters(p["k"], p["c"], c_m, ic)

                    # instantiate PS with correct settings
                    ps = ParticleSystem(c_m, ic, e_m, p)

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
            print(f"PS kinq damp: {times[2]:.3f}")

