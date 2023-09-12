"""
Additional package SystemEnergy, which calculates total PS energy:
    gravitational potential + kinetic energy + spring potential - energy dissipation by internal friction
"""
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem
import numpy as np
import numpy.typing as npt
from numpy import linalg as la


def system_energy(ps: ParticleSystem, params: dict, v_prev: npt.ArrayLike = np.array([0, 0, 0])):

    k = params['k']
    c = params['c']
    g = params["g"]
    n = params["n"]
    dt = params["dt"]
    l0 = params["l0"]

    particles = ps.particles

    # Elastic potential
    ep = np.zeros(n - 1,)
    for i in range(n - 1):      # loops over springs, not particles
        ep[i] = 0.5 * k * (la.norm(particles[i].x - particles[i + 1].x) - l0) ** 2

    # Kinetic energy
    ke = np.array([0.5 * la.norm(particle.v) ** 2 for particle in particles])

    # Gravitational potential
    gp = np.array([particle.m * g * particle.x[-1] for particle in particles])

    # # Energy dissipated by friction
    # ed = np.array([c * dt * abs(la.norm(particles[i].v)**2 - la.norm(v_prev[i*3:(i + 1)*3])**2) for i in range(n)])

    # Total system energy
    te = sum(ep) + sum(ke) + sum(gp) #- sum(ed)

    # print("ep:", ep)
    # print("ke:", ke)
    # print("gp:", gp)
    # print("ed:", ed)
    # print("te:", te)
    return te

