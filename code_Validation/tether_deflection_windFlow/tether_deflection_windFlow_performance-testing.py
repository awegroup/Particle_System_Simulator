"""
Script for PS framework validation, benchmark case where tether is fixed at both ends and is deflected by perpendicular
wind flow
"""
import numpy as np
import tether_deflection_windFlow_input as input
from tether_deflection_windFlow_input import params as p
import timeit
import time
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem
np.seterr(all='raise')


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, p)


def calculate_f_a(ps: ParticleSystem):
    particle_list = ps.particles
    f_a = np.zeros(p['n']*3, )
    rho = 1.225

    for i in range(len(particle_list) - 1):
        V_b = 0.5 * (particle_list[i].v + particle_list[i + 1].v)  # velocity of the bridle = avg vel. of the particles
        V_b_app = input.params["v_w"] - V_b  # apparent velocity of bridle
        V_b_norm = np.linalg.norm(V_b_app)
        x = particle_list[i].x - particle_list[i + 1].x
        l_element = np.linalg.norm(x)

        # derivation of equation below, see "Bridle Particle pdf"
        S_eff_bridle = input.params["d_bridle"] * np.sqrt(l_element ** 2 - (np.dot(V_b_app, x) / V_b_norm) ** 2)
        F_a_drag = 0.5 * rho * V_b_app * V_b_norm * S_eff_bridle * input.params['c_d_bridle']

        # Drag force, includes the direction of the velocity
        f_a[i * 3:(i + 1) * 3] += 0.5 * F_a_drag
        f_a[(i + 1) * 3:(i + 2) * 3] += 0.5 * F_a_drag

    return f_a


def converge(psystem: ParticleSystem, c: int):
    c_cond = False
    try:
        if c == 1:
            for i in range(p['t_steps']):           # viscous damping
                f_aero = calculate_f_a(psystem)
                psystem.simulate(f_aero)
                f_aero[0:3] = 0
                f_aero[-3:] = 0
                residual_f = f_aero + psystem.f_int
                if np.linalg.norm(residual_f) <= 1e-3:
                    c_cond = True
                    break

        elif c == 2:
            for i in range(p['t_steps']):           # kin damping without q
                f_aero = calculate_f_a(psystem)
                psystem.kin_damp_sim(f_aero)
                f_aero[0:3] = 0
                f_aero[-3:] = 0
                residual_f = f_aero + psystem.f_int
                if np.linalg.norm(residual_f) <= 1e-3:
                    c_cond = True
                    break

        elif c == 3:
            for i in range(p['t_steps']):           # kin damping with q
                f_aero = calculate_f_a(psystem)
                psystem.kin_damp_sim(f_aero, q_correction=True)
                f_aero[0:3] = 0
                f_aero[-3:] = 0
                residual_f = f_aero + psystem.f_int
                if np.linalg.norm(residual_f) <= 1e-3:
                    c_cond = True
                    break
    except FloatingPointError:
        print('unstable sim encountered')

    if not c_cond:
        print("convergence not reached, option:", c)
    return


if __name__ == "__main__":
    # result settings
    times = np.zeros((3, ))
    samplesize = 10
    n = [13, 41, 85, 181]#, 761]
    # n = [181]
    # n = [761]
    k = [1, 100, 6e4, 1e6]
    # k = [1e6]
    p['m'] = 1
    p['c'] = 0
    p['dt'] = 1
    p['t_steps'] = 1000

    for stiffness in k:
        p['k_t'] = stiffness
        for amount in n:
            print()
            p['n'] = amount

            # looping simulations
            time1 = time.time()
            for i in range(1, 3):
                for j in range(samplesize):
                    print(i, j)
                    # recalculate parameters
                    p["l0"] = p["L"] / (p["n"] - 1)
                    # p["m_segment"] = p["L"] * p["rho_tether"] / (p["n"] - 1)
                    p["k"] = p["k_t"] * (p["n"] - 1)
                    c_m = input.connectivity_matrix(amount)
                    # ic = input.initial_conditions(p['l0'], amount, p['m_segment'])
                    ic = input.initial_conditions(p['l0'], amount, p['m'])

                    # instantiate PS with settings
                    ps = ParticleSystem(c_m, ic, p)
                    # print(p)
                    # measure convergence time
                    times[i-1] += timeit.repeat(lambda: converge(ps, i), repeat=1, number=1)
            time2 = time.time()
            print(f'total loop time: {(time2 - time1):.4f} s')

            # printing statistics
            times = times/samplesize
            print(f"input settings: h = {p['dt']}, n = {p['n']}, k = {p['k_t']}, c = {p['c']}")
            print("average runtimes")
            print(f"PS vis damp: {times[0]:.3f}")
            print(f"PS kin damp: {times[1]:.3f}")
            print(f"PS kinq damp: {times[2]:.3f}")

