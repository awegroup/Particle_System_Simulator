"""
Script for verification of correct implementation spring force of SpringDamper object within ParticleSystem framework
"""
import numpy
import numpy as np
from scipy.sparse.linalg import bicgstab
import input_damping_force as input
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

# adjusting directory where modules are imported from
import sys

sys.path.insert(1, sys.path[0][:-31] + 'src/ParticleSystem')
from ParticleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.params)


def simulate(psystem: ParticleSystem):  # propagates simulation one timestep
    dt = input.params["dt"]
    rtol = input.params["rel_tol"]
    atol = input.params["abs_tol"]
    maxiter = input.params["max_iter"]

    mass_matrix = psystem.m_matrix()
    f = psystem.one_d_force_vector()
    v_current = psystem.pack_v_current()
    x_current = psystem.pack_x_current()

    jx, jv = psystem.system_jacobians()

    # constructing A matrix and b vector for solver
    A = mass_matrix - dt * jv - dt ** 2 * jx
    b = dt * f + dt ** 2 * np.matmul(jx, v_current)

    # BiCGSTAB from scipy library
    dv, _ = bicgstab(A, b, tol=rtol, atol=atol, maxiter=maxiter)
    v_next = v_current + dv
    x_next = x_current + dt * v_next

    psystem.update_x_v(x_next, v_next)

    return x_next[-1], v_next[-1]


def exact_solution():
    k = input.params["k"]
    c = input.params["c"]
    m = input.init_cond[1][-2]
    omega = np.sqrt(k / m)
    gamma = c/m
    zeta = c/(2 * omega)        # critical damping faction

    # exact solution depends on value of zeta
    if zeta <1:
        print("system is underdamped")
    elif zeta == 1:
        print("system is critically damped")
    else:
        print("system is overdamped")

    # solve as IVP
    y0 = np.array([1, 0])

    def syst_of_diff_eq(t, y):
        A = np.array([[0, 1], [-omega**2, -gamma]])
        system = np.matmul(A, y)
        return system

    t_vector = np.linspace(0, input.params["t_steps"] * input.params["dt"], input.params["t_steps"] + 1)

    t_int = [t_vector[0], t_vector[-1]]
    ivp_solution = solve_ivp(syst_of_diff_eq, t_int, y0=y0, t_eval=t_vector)

    # # exact solution
    # c1 = 0.5
    # c2 = 0.5
    # exact_x = np.exp(-.5 * c * t_vector) * (c1 * np.exp(t_vector * np.sqrt((c/2)**2 - omega**2)
    #                                                     + c2 * np.exp(-t_vector * np.sqrt((c/2)**2 - omega**2))))

    # Estimated (expected) decay rate of implicit Euler scheme as a function of t
    dt = input.params['dt']
    decay = np.exp(-0.5 * omega ** 2 * dt * t_vector)

    return t_vector, ivp_solution, decay


def plot(psystem: ParticleSystem):
    t_vector = np.linspace(input.params["dt"], input.params["t_steps"] * input.params["dt"], input.params["t_steps"])
    x = {"x": np.zeros(len(t_vector), )}
    v = {"v": np.zeros(len(t_vector), )}

    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)

    position.iloc[0] = input.init_cond[1][0][-1]
    velocity.iloc[0] = input.init_cond[1][1][-1]

    for step in t_vector:
        position.loc[step], velocity.loc[step] = simulate(psystem)

    t, exact, decay = exact_solution()
    print(position["x"])
    print()
    print(len(position["x"]), len(decay[1:]))

    corrected = numpy.divide(np.array(position["x"]), decay[1:])

    # graph configuration
    position.plot()
    plt.plot(t, exact.y[0])
    plt.plot(t, decay)
    plt.plot(t[1:], corrected)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Verification PS damping force implementation with Implicit Euler scheme")
    plt.legend(["PS simulation", "Exact solution", "Decay rate implicit Euler", "corrected for decay rate"])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    file_path = sys.path[0][:-31] + "code_Verification/verification_results/damping_force/"
    img_name = f"{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['c']}dampingC-" \
               f"{input.params['dt']}timestep.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()

    return


if __name__ == "__main__":
    ps = instantiate_ps()

    plot(ps)

    """
    Notes for self: Try to plot decay rate and clean up code before committing and sending. 
                    Add clause for (zero) spring rest length to SpringDamper

    today: cleanup code, 
           added automated result generation, estimated decay and phase shift, gravitational force?
    """