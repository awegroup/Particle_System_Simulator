"""
ParticleSystem framework
...
"""
import numpy as np
import numpy.typing as npt
from src.particleSystem.Particle import Particle
from src.particleSystem.SpringDamper import SpringDamper 
from scipy.sparse.linalg import bicgstab


class ParticleSystem:
    def __init__(self, 
                 connectivity_matrix: list, 
                 initial_conditions: npt.ArrayLike, 
                 sim_param: dict):
        """
        Constructor for ParticleSystem object, model made up of n particles
        :param connectivity_matrix: 2-by-m matrix, where each column contains a nodal index pair that is connected
                                    by a spring element.
        :param initial_conditions:  Array of n arrays to instantiate particles. Each subarray must contain the params
                                    required for the particle constructor: [initial_pos, initial_vel, mass, fixed: bool]
        :param element_params:      Array of m arrays to instantiate elements. Each subarray must contain the remaining
                                    params required for the element constructor: [k, l0, c, compressive_resistant, ...]
                                    # note: could change depending on what element types are added in the future.
        :param sim_param:           Dictionary of other parameters required for simulation (dt, rtol, ...)
        """
        self.__connectivity_matrix = connectivity_matrix

        self.__n = len(initial_conditions)
        self.__dt = sim_param["dt"]
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

        # Variables required for kinetic damping
        self.__w_kin = self.__calc_kin_energy()
        self.__w_kin_min1 = self.__calc_kin_energy()
        self.__vis_damp = True
        self.__x_min1 = np.zeros(self.__n, )
        self.__x_min2 = np.zeros(self.__n, )
        return

    def __str__(self):
        print("ParticleSystem object instantiated with attributes\nConnectivity matrix:")
        print(self.__connectivity_matrix)
        print("Instantiated particles:")
        n = 1
        for particle in self.__particles:
            print(f" p{n}: ", particle)
            n += 1
        return ""

    def __instantiate_particles(self, initial_conditions: list):
        for set_of_initial_cond in initial_conditions:
            x = set_of_initial_cond[0]
            v = set_of_initial_cond[1]
            m = set_of_initial_cond[2]
            f = set_of_initial_cond[3]
            self.__particles.append(Particle(x, v, m, f))
        return

    def __instantiate_springdampers(self):
        for link in self.__connectivity_matrix:
            link = link.copy() #needed to not override the __connectivity_matrix
            link[0] = self.__particles[link[0]]
            link[1] = self.__particles[link[1]]
            self.__springdampers.append(SpringDamper(*link))
        return
    
    def stress_self(self, factor: float = 0):
        """Set all node lengths to zero to homogenously stress mesh"""
        if factor == 0:
            for link in self.springdampers:
                link.l0 = 0
        else:
            for link in self.springdampers:
                link.l0 *= factor 
                
        return
        

    def __construct_m_matrix(self):
        matrix = np.zeros((self.__n * 3, self.__n * 3))

        for i in range(self.__n):
            matrix[i*3:i*3+3, i*3:i*3+3] += np.identity(3)*self.__particles[i].m

        return matrix

    def __calc_kin_energy(self):
        v = self.__pack_v_current()
        w_kin = np.matmul(np.matmul(v, self.__m_matrix), v.T)      # Kinetic energy, 0.5 constant can be neglected
        return w_kin

    def simulate(self, f_external: npt.ArrayLike = ()):
        if not len(f_external):             # check if external force is passed as argument, otherwise use 0 vector
            f_external = np.zeros(self.__n * 3, )
        f = self.__one_d_force_vector() + f_external

        v_current = self.__pack_v_current()
        x_current = self.__pack_x_current()

        jx, jv = self.__system_jacobians()

        # constructing A matrix and b vector for solver
        A = self.__m_matrix - self.__dt * jv - self.__dt ** 2 * jx
        b = self.__dt * f + self.__dt ** 2 * np.matmul(jx, v_current)

        # checking conditioning of A
        # print("conditioning A:", np.linalg.cond(A))

        for i in range(self.__n):
            if self.__particles[i].fixed:
                A[i * 3: (i + 1) * 3] = 0        # zeroes out row i to i + 3
                A[:, i * 3: (i + 1) * 3] = 0     # zeroes out column i to i + 3
                b[i * 3: (i + 1) * 3] = 0        # zeroes out row i

        # BiCGSTAB from scipy library
        dv, _ = bicgstab(A, b, tol=self.__rtol, atol=self.__atol, maxiter=self.__maxiter)

        # numerical time integration following implicit Euler scheme
        v_next = v_current + dv
        x_next = x_current + self.__dt * v_next

        # function returns the pos. and vel. for the next timestep, but for fixed particles this value doesn't update!
        self.__update_x_v(x_next, v_next)
        return x_next, v_next

    def kin_damp_sim(self, f_ext: npt.ArrayLike = (), q_correction: bool = False):       # kinetic damping algorithm
        if self.__vis_damp:         # Condition resetting viscous damping to 0
            for link in self.__springdampers:
                link.c = 0
            self.__c = 0
            self.__vis_damp = False

        if len(f_ext):              # condition checking if an f_ext is passed as argument
            self.__save_state()
            x_next, v_next = self.simulate(f_ext)
        else:
            self.__save_state()
            x_next, v_next = self.simulate()

        w_kin_new = self.__calc_kin_energy()

        if w_kin_new > self.__w_kin:    # kin damping algorithm, takes effect when decrease in kin energy is detected
            self.__update_w_kin(w_kin_new)
        else:
            v_next = np.zeros(self.__n*3, )

            if q_correction:            # statement to check if q_correction is desired, standard is turned off
                q = (self.__w_kin - w_kin_new)/(2*self.__w_kin - self.__w_kin_min1 - w_kin_new)
                # print(q)
                # print(self.__w_kin, w_kin_new)
                # !!! Not sure if linear interpolation between states is the way to determine new x_next !!!
                if q < 0.5:
                    x_next = self.__x_min2 + (q / 0.5) * (self.__x_min1 - self.__x_min2)
                elif q == 0.5:
                    x_next = self.__x_min1
                elif q < 1:
                    x_next = self.__x_min1 + ((q - 0.5) / 0.5) * (x_next - self.__x_min1)

                # Can also use this q factor to recalculate the state for certain timestep h

            self.__update_x_v(x_next, v_next)
            self.__update_w_kin(0)

        return x_next, v_next

    def __pack_v_current(self):
        return np.array([particle.v for particle in self.__particles]).flatten()

    def __pack_x_current(self):
        return np.array([particle.x for particle in self.__particles]).flatten()

    def __one_d_force_vector(self):
        self.__f[self.__f != 0] = 0

        for n in range(len(self.__springdampers)):
            f_int = self.__springdampers[n].force_value()
            i, j, *_ = self.__connectivity_matrix[n]

            self.__f[i*3: i*3 + 3] += f_int
            self.__f[j*3: j*3 + 3] -= f_int

        return self.__f

    def __system_jacobians(self):
        self.__jx[self.__jx != 0] = 0
        self.__jv[self.__jv != 0] = 0

        for n in range(len(self.__springdampers)):
            jx, jv = self.__springdampers[n].calculate_jacobian()
            i, j, *_ = self.__connectivity_matrix[n]

            self.__jx[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jx
            self.__jx[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jx
            self.__jx[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jx
            self.__jx[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jx

            self.__jv[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jv
            self.__jv[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jv
            self.__jv[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jv
            self.__jv[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jv

        return self.__jx, self.__jv

    def __update_x_v(self, x_next: npt.ArrayLike, v_next: npt.ArrayLike):
        for i in range(self.__n):
            self.__particles[i].update_pos(x_next[i * 3:i * 3 + 3])
            self.__particles[i].update_vel(v_next[i * 3:i * 3 + 3])
        return

    def __update_w_kin(self, w_kin_new: float):
        self.__w_kin_min1 = self.__w_kin
        self.__w_kin = w_kin_new
        return

    def __save_state(self):
        self.__x_min2 = self.__x_min1
        self.__x_min1 = self.__pack_x_current()
        return

    @property
    def particles(self):            # @property decorators required, as PS info might be required for external calcs
        return self.__particles

    @property
    def springdampers(self):
        return self.__springdampers

    # @property
    # def stiffness_m(self):
    #     self.__system_jacobians()
    #     return self.__jx

    @property
    def f_int(self):
        f_int = self.__f.copy()
        for i in range(len(self.__particles)):      # need to exclude fixed particles for force-based convergence
            if self.__particles[i].fixed:
                f_int[i*3:(i+1)*3] = 0

        return f_int

    @property
    def x_v_current(self):
        return self.__pack_x_current(), self.__pack_v_current()


if __name__ == "__main__":
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
        "g": 9.81           # [m/s^2] gravitational acceleration
    }
    c_matrix = [[0, 1, params['k'], params['c'] ]]
    init_cond = [[[0, 0, 0], [0, 0, 0], 1, True],
                 [[0, 0, 0], [0, 0, 0], 1, False]]


    ps = ParticleSystem(c_matrix, init_cond, params)
    print(ps)
    pass
