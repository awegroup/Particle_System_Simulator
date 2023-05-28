"""
Script for verification of correct implementation external gravitational force within ParticleSystem framework
"""
import numpy as np
from scipy.sparse.linalg import bicgstab
import input_gravitational_force as input
import matplotlib.pyplot as plt
import pandas as pd

# adjusting directory where modules are imported from
import sys
sys.path.insert(1, sys.path[0][:-37] + 'src/ParticleSystem')
from ParticleSystem import ParticleSystem


def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.params)


def external_forces(n: int, g: float):
    return np.array([[0, 0, -g] for i in range(n)]).flatten()


def simulate(psystem: ParticleSystem):  # propagates simulation one timestep
    dt = input.params["dt"]
    rtol = input.params["rel_tol"]
    atol = input.params["abs_tol"]
    maxiter = input.params["max_iter"]
    g = input.params["g"]
    n = input.params["n"]

    mass_matrix = psystem.m_matrix()
    f = psystem.one_d_force_vector()
    f_ext = external_forces(n, g)
    f += f_ext

    v_current = psystem.pack_v_current()
    x_current = psystem.pack_x_current()

    jx, jv = psystem.system_jacobians()

    jx = jv = np.zeros((2 * 3, 2 * 3))  # damping Jacobian set to zero, no damping included

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
    g = input.params["g"]
    m = input.init_cond[1][-2]
    t_vector = np.linspace(0, input.params["t_steps"] * input.params["dt"], input.params["t_steps"] + 1)

    exact_x = -0.5 * g * t_vector**2

    return t_vector, exact_x


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

    t, exact = exact_solution()

    # graph configuration
    position.plot()
    plt.plot(t, exact)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Verification external gravitational force")
    plt.legend(["PS simulation", "Exact solution"])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    file_path = sys.path[0][:-37] + "code_Verification/verification_results/gravitational_force/"
    img_name = f"{input.params['n']}Particles-{input.params['dt']}timestep.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()

    return


if __name__ == "__main__":
    ps = instantiate_ps()

    plot(ps)
