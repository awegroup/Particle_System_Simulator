"""
Script for verification of correct implementation spring force of SpringDamper object within ParticleSystem framework
"""
import numpy as np
from scipy.sparse.linalg import bicgstab
import input_spring_force as input


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
    print(v_current)

    jx = psystem.system_jacobian()          # only returns spring Jacobian for now
    # print(jx)

    jv = np.zeros((2 * 3, 2 * 3))           # damping Jacobian set to zero should work

    # constructing A matrix and b vector for solver
    A = mass_matrix - dt*jv - dt**2*jx
    b = dt*f + dt**2*np.matmul(jx, v_current)


    dv, _ = bicgstab(A, b, tol=rtol, atol=atol, maxiter=maxiter)
    print(dv)

    v_next = v_current + dt*dv
    print(v_next)

    return

def exact_solution():

    return

def plot():

    return


if __name__ == "__main__":

    ps = instantiate_ps()
    # print(ps)

    simulate(ps)
