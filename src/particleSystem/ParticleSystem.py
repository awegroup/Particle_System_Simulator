"""
ParticleSystem framework
...
"""
import numpy as np
import numpy.typing as npt
from Msc_Alexander_Batchelor.src.particleSystem.Particle import Particle
from Msc_Alexander_Batchelor.src.particleSystem.SpringDamper import SpringDamper
from scipy.sparse.linalg import bicgstab


class ParticleSystem:
    def __init__(self, connectivity_matrix: npt.ArrayLike, initial_conditions: npt.ArrayLike,
                 sim_param: dict):
        """
        Constructor for ParticleSystem object, model made up of n particles
        :param connectivity_matrix: sparse n-by-n matrix, where an 1 at index (i,j) means
                                    that particle i and j are connected
        :param initial_conditions: Array of n arrays to instantiate particles. Each array must contain the information
                                   required for the particle constructor: [initial_pos, initial_vel, mass, fixed: bool]
        :param sim_param: Dictionary of other parameters required (k, l0, dt, ...)
        """
        self.__connectivity_matrix = np.array(connectivity_matrix)
        self.__k = sim_param["k"]
        self.__l0 = sim_param["l0"]
        self.__c = sim_param["c"]
        self.__dt = sim_param["dt"]
        self.__n = sim_param["n"]

        self.__rtol = sim_param["rel_tol"]
        self.__atol = sim_param["abs_tol"]
        self.__maxiter = sim_param["max_iter"]

        # allocate memory
        self.__particles = []
        self.__springdampers = []
        self.__f = np.zeros((self.__n * 3, ))
        self.__jx = np.zeros((self.__n * 3, self.__n * 3))
        self.__jv = np.zeros((self.__n * 3, self.__n * 3))

        self.__instantiate_particles(initial_conditions)
        self.__m_matrix = self.__construct_m_matrix()
        self.__instantiate_springdampers()
        return

    def __str__(self):
        print("ParticleSystem object instantiated with attributes\nConnectivity matrix:")
        print(self.__connectivity_matrix)
        print("Instantiated particles:")
        n = 1
        for particle in self.__particles:
            print(f" p{n}: ", particle)
            n += 1
        # print("Instantiated spring-damper elements:")
        # for springdamper in self.__springdampers:
        #     print(" ", springdamper)
        return ""

    def __instantiate_particles(self, initial_conditions):
        for set_of_initial_cond in initial_conditions:
            x = set_of_initial_cond[0]
            v = set_of_initial_cond[1]
            m = set_of_initial_cond[2]
            f = set_of_initial_cond[3]
            self.__particles.append(Particle(x, v, m, f))
        return

    def __instantiate_springdampers(self):
        b = np.nonzero(np.triu(self.__connectivity_matrix))
        b = np.column_stack((b[0], b[1]))
        for index in b:
            self.__springdampers.append(SpringDamper(self.__particles[index[0]], self.__particles[index[1]],
                                        self.__k, self.__l0, self.__c, self.__dt))
        return

    def __construct_m_matrix(self):
        matrix = np.zeros((self.__n * 3, self.__n * 3))

        for i in range(self.__n):
            matrix[i*3:i*3+3, i*3:i*3+3] += np.identity(3)*self.__particles[i].m

        return matrix

    def simulate(self, f_external: npt.ArrayLike = False):
        if not f_external.any():
            f_external = np.zeros(self.__n * 3, )
        f = self.__one_d_force_vector() + f_external
        v_current = self.__pack_v_current()
        x_current = self.__pack_x_current()

        jx, jv = self.__system_jacobians()

        # constructing A matrix and b vector for solver
        A = self.__m_matrix - self.__dt * jv - self.__dt ** 2 * jx
        b = self.__dt * f + self.__dt ** 2 * np.matmul(jx, v_current)

        # BiCGSTAB from scipy library
        dv, _ = bicgstab(A, b, tol=self.__rtol, atol=self.__atol, maxiter=self.__maxiter)
        v_next = v_current + dv
        x_next = x_current + self.__dt * v_next

        # function returns the pos. and vel. for the next timestep, but for fixed particles this value doesn't update!
        self.__update_x_v(x_next, v_next)
        return x_next, v_next

    def __pack_v_current(self):
        return np.array([particle.v for particle in self.__particles]).flatten()

    def __pack_x_current(self):
        return np.array([particle.x for particle in self.__particles]).flatten()

    def __one_d_force_vector(self):
        self.__f[self.__f != 0] = 0

        for i in range(self.__n - 1):
            fs, fd = self.__springdampers[i].force_value()
            self.__f[i*3: i*3 + 3] += fs + fd
            self.__f[(i + 1)*3: (i + 1)*3 + 3] -= fs + fd

        return self.__f

    def __system_jacobians(self):
        self.__jx[self.__jx != 0] = 0
        self.__jv[self.__jv != 0] = 0

        i = 0
        j = 1
        for springdamper in self.__springdampers:
            jx, jv = springdamper.calculate_jacobian()
            # print(springdamper)

            # self.__jx[i*3:i*3+3, i*3:i*3+3] += jx
            # self.__jx[j*3:j*3+3, j*3:j*3+3] += jx
            # self.__jx[i*3:i*3+3, j*3:j*3+3] -= jx
            # self.__jx[j*3:j*3+3, i*3:i*3+3] -= jx

            self.__jx[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jx
            self.__jx[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jx
            self.__jx[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jx
            self.__jx[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jx

            self.__jv[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jv
            self.__jv[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jv
            self.__jv[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jv
            self.__jv[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jv

            i += 1
            j += 1

        return self.__jx, self.__jv

    def __update_x_v(self, x_next: npt.ArrayLike, v_next: npt.ArrayLike):
        for i in range(self.__n):
            self.__particles[i].update_pos(x_next[i * 3:i * 3 + 3])
            self.__particles[i].update_vel(v_next[i * 3:i * 3 + 3])
        return

    @property
    def particles(self):            # Temporary solution to calculate external aerodynamic forces
        return self.__particles


if __name__ == "__main__":

    c_matrix = [[0, 1], [1, 0]]
    init_cond = [[[0, 0, 0], [0, 0, 0], 1, True], [[0, 0, 0], [0, 0, 0], 1, False]]

    params = {
        # model parameters
        "n": 2,  # [-] number of particles
        "k": 2e4,  # [N/m] spring stiffness
        "c": 0,  # [N s/m] damping coefficient
        "l0": 0,  # [m] rest length

        # simulation settings
        "dt": 0.001,  # [s] simulation timestep
        "t_steps": 1000,  # [-] number of simulated time steps
        "abs_tol": 1e-50,  # [m/s] absolute error tolerance iterative solver
        "rel_tol": 1e-5,  # [-] relative error tolerance iterative solver
        "max_iter": 1e5,  # [-] maximum number of iterations

        # physical parameters

    }

    ps = ParticleSystem(c_matrix, init_cond, params)
    print(ps)
    pass
