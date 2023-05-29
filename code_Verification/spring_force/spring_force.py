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


def simulate(psystem: ParticleSystem):      # propagates simulation one timestep
    dt = input.params["dt"]
    rtol = input.params["rel_tol"]
    atol = input.params["abs_tol"]
    maxiter = input.params["max_iter"]

    mass_matrix = psystem.m_matrix()
    f = psystem.one_d_force_vector()
    v_current = psystem.pack_v_current()
    x_current = psystem.pack_x_current()

    jx, jv = psystem.system_jacobians()

    jv = np.zeros((2 * 3, 2 * 3))           # damping Jacobian set to zero, no damping included

    # constructing A matrix and b vector for solver
    A = mass_matrix - dt*jv - dt**2*jx
    b = dt*f + dt**2*np.matmul(jx, v_current)

    # BiCGSTAB from scipy library
    dv, _ = bicgstab(A, b, tol=rtol, atol=atol, maxiter=maxiter)
    v_next = v_current + dv
    x_next = x_current + dt*v_next

    psystem.update_x_v(x_next, v_next)

    return x_next[-1], v_next[-1]


def exact_solution():
    k = input.params["k"]
    m = input.init_cond[1][-2]
    omega = np.sqrt(k/m)
    t_vector = np.linspace(0, input.params["t_steps"]*input.params["dt"], input.params["t_steps"]+1)

    exact_x = np.cos(omega * t_vector)

    # Estimated (expected) decay rate of implicit Euler scheme as a function of t
    dt = input.params['dt']
    decay = np.exp(-0.5 * omega**2 * dt * t_vector)

    # # Estimated (expected) phase shift (freq. reduction) of implicit Euler scheme as a function of t
    # import cmath
    # real = np.zeros((len(t_vector,)))
    # phase_shift = np.exp(-omega * t_vector * (1 - 1/3 * (omega * dt)**2))
    # imaginary_vector = list(zip(real, phase_shift))
    # shift = cmath.phase(complex(imaginary_vector[-1][0], imaginary_vector[-1][1]))
    # print(f"Estimated phase shift: {shift:.3f} pi")
    # NOT ACCURATE, NEEDS MORE THINKING TROUGH

    return t_vector, exact_x, decay


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

    t, exact, decay = exact_solution()

    # graph configuration
    position.plot()
    plt.plot(t, exact)
    plt.plot(t, decay)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Verification PS spring force implementation with Implicit Euler scheme")
    plt.legend(["PS simulation", "Exact solution", "Decay rate implicit Euler"])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    file_path = sys.path[0][:-30] + "code_Verification/verification_results/spring_force/"
    img_name = f"{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['dt']}timestep.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()

    return


if __name__ == "__main__":

    ps = instantiate_ps()

    plot(ps)


    """
    Notes for self: Longitudal oscillations benchmark analytical sol.
                    Make sure every verification runs w/o error and clean up before committing and sending. 
                    Check verification gravitational force
                    Make function for system energy
                    Make plot gravity verification abs. and rel. error
                    Make plot for internal force values
                    
    today:          Started work on long. oscillations benchmark
           
    """