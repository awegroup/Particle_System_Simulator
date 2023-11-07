"""
Script for PS framework validation, benchmark case where tether is fixed at both ends and is deflected by perpendicular
wind flow
"""
import matplotlib.pyplot as plt
import numpy as np
import tether_deflection_gravity_input as input
from tether_deflection_gravity_input import params as p
import numpy.typing as npt
import timeit
import time
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.elem_params, p)


def converge(psystem: ParticleSystem, c: int, f_ext: npt.ArrayLike):
    f_test = f_ext.copy()
    f_test[0:3] = 0
    f_test[-2:] = 0
    cond = False

    if c == 1:
        for m in range(p['t_steps']):           # viscous damping
            psystem.simulate(f_ext)
            residual_f = f_test + psystem.f_int
            if np.linalg.norm(residual_f) <= 1e-3:
                cond = True
                break

    elif c == 2:
        for m in range(p['t_steps']):  # kin damping without q
            psystem.kin_damp_sim(f_ext)
            residual_f = f_test + psystem.f_int
            if np.linalg.norm(residual_f) <= 1e-3:
                cond = True
                break

    elif c == 3:
        for m in range(p['t_steps']):  # kin damping with q
            x_step, _ = psystem.kin_damp_sim(f_ext, q_correction=True)
            residual_f = f_test + psystem.f_int
            if np.linalg.norm(residual_f) <= 1e-3:
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
    n = [13, 41, 85, 181]#, 761]
    # n = [761]
    k = [1, 100, 6e4, 1e6]

    p['m'] = 1
    p['c'] = 1
    p['dt'] = 1
    p['t_steps'] = 1000

    for stiffness in k:
        for amount in n:
            print()
            p['n'] = amount
            # looping simulations
            time1 = time.time()
            for i in range(1, 3):
                for j in range(samplesize):
                    print(i, j)
                    # recalculate parameters
                    p["l0"] = p["L"] / (amount - 1)
                    # p["m_segment"] = p["L"] * p["rho_tether"] / (p["n"] - 1)
                    p["k"] = stiffness * (amount - 1)
                    c_m = input.connectivity_matrix(amount)
                    # ic = input.initial_conditions(p['l0'], amount, p['m_segment'])
                    ic = input.initial_conditions(p['l0'], amount, p['m'])
                    e_m = input.element_parameters(p["k"], p["l0"], p["c"], p["n"])

                    # instantiate PS with settings
                    ps = ParticleSystem(c_m, ic, e_m, p)

                    particles = ps.particles
                    f_ext = np.array([[0, 0, -p['g'] * particle.m] for particle in particles]).flatten()

                    # measure convergence time
                    times[i-1] += timeit.repeat(lambda: converge(ps, i, f_ext), repeat=1, number=1)
            time2 = time.time()
            print(f'total loop time: {(time2 - time1):.4f} s')

            # printing statistics
            times = times/samplesize
            print(f"input settings: h = {p['dt']}, n = {p['n']}, k = {p['k']}, c = {p['c']}")
            print("average runtimes")
            print(f"PS vis damp: {times[0]:.3f}")
            print(f"PS kin damp: {times[1]:.3f}")
            print(f"PS kinq damp: {times[2]:.3f}")

