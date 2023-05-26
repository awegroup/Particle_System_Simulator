"""
Script for verification of correct implementation spring force of SpringDamper object within ParticleSystem framework
"""
import numpy as np
from scipy.sparse.linalg import bicgstab
import input_spring_force as input
import matplotlib.pyplot as plt
import pandas as pd

# adjusting directory where modules are imported from
import sys
sys.path.insert(1, sys.path[0][:-30]+'src/ParticleSystem')
from ParticleSystem import ParticleSystem


def instantiate_ps():

    return ParticleSystem(input.c_matrix, input.init_cond, input.params)


def simulate(psystem: ParticleSystem):
    dt = input.params["dt"]
    rtol = input.params["rel_tol"]
    atol = input.params["abs_tol"]
    maxiter = input.params["max_iter"]

    mass_matrix = psystem.m_matrix()
    # print(mass_matrix)

    f = psystem.one_d_force_vector()
    # print(f)

    v_current = psystem.pack_v_current()
    x_current = psystem.pack_x_current()

    # print()
    # print("v_current: ", v_current[-3:])
    # print("x_current: ", x_current[-3:])

    jx = psystem.system_jacobian()          # only returns spring Jacobian for now
    # print(jx)

    jv = np.zeros((2 * 3, 2 * 3))           # damping Jacobian set to zero should work

    # constructing A matrix and b vector for solver
    A = mass_matrix - dt*jv - dt**2*jx
    b = dt*f + dt**2*np.matmul(jx, v_current)

    # BiCGSTAB from scipy library
    dv, _ = bicgstab(A, b, tol=rtol, atol=atol, maxiter=maxiter)
    v_next = v_current + dv
    x_next = x_current + dt*v_next

    # print()
    # print("delta_v: ", dv[-3:])
    # print("v_next: ", v_next[-3:])
    # print("x_next: ", x_next[-3:])

    # i = np.identity(2*3)
    # m_inv = np.linalg.inv(mass_matrix)
    # A = i - dt * np.matmul(m_inv, jv) - dt**2 * np.matmul(m_inv, jx)
    # b = dt * np.matmul(m_inv, f + dt * np.matmul(jx, v_current))
    #
    # # BiCGSTAB from scipy library
    # dv, _ = bicgstab(A, b, tol=rtol, atol=atol, maxiter=maxiter)
    # print()
    # print("delta_v: ", dv[-3:])
    #
    # v_next = v_current + dv
    # print("v_next: ", v_next[-3:])
    #
    # x_next = x_current + dt*v_next
    # print("x_next: ", x_next[-3:])

    psystem.update_x_v(x_next, v_next)

    return x_next[-1], v_next[-1]


def exact_solution():
    k = input.params["k"]
    m = input.init_cond[1][-2]
    omega = np.sqrt(k/m)      # conversion to rad
    t_vector = np.linspace(0, input.params["t_steps"]*input.params["dt"], input.params["t_steps"]+1)
    exact_x = np.cos(omega * t_vector )

    # exact_v = -omega * np.sin(omega*t_vector)
    # exact_a = -omega ** 2 * np.cos(omega*t_vector)
    #
    # print()
    # print("time :", t_vector)
    # print("pos: ", exact_x)
    # print("vel: ", exact_v)
    # print("acc: ", exact_a)

    # plt.plot(t_vector, exact_x)
    # plt.plot(t_vector, exact_v)
    # plt.plot(t_vector, exact_a)
    # plt.legend()
    # plt.grid()
    # plt.show()
    return t_vector, exact_x


def plot(psystem: ParticleSystem):
    t_vector = np.linspace(input.params["dt"], input.params["t_steps"] * input.params["dt"], input.params["t_steps"])
    x = {"x": np.zeros(len(t_vector),)}
    v = {"v": np.zeros(len(t_vector), )}

    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)

    position.iloc[0] = input.init_cond[1][0][-1]
    velocity.iloc[0] = input.init_cond[1][1][-1]

    for step in t_vector:
        position.loc[step], velocity.loc[step] = simulate(psystem)

    t, exact = exact_solution()

    position.plot()
    plt.plot(t, exact)
    plt.grid()
    plt.show()

    print(position)
    print(velocity)

    return


if __name__ == "__main__":

    ps = instantiate_ps()
    # print(ps)

    # simulate(ps)
    # print(ps)

    plot(ps)

    # exact_solution()
