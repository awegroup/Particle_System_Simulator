"""
Child Class 'SpringDamper', for spring-damper objects to be instantiated in ParticleSystem
"""
from ImplicitForce import ImplicitForce
from Particle import Particle
import numpy as np


class SpringDamper(ImplicitForce):

    def __init__(self, p1: Particle, p2: Particle, k: float, l0: float, dt: float):
        self.__k = k
        self.__l0 = l0
        self.__dt = dt
        super().__init__(p1, p2)
        return

    def __str__(self):
        return f"SpringDamper object, spring stiffness [n/m]: {self.__k}, rest length [m]: {self.__l0}\n" \
               f"Assigned particles\n  p1: {self.p1}\n  p2: {self.p2}"

    def __relative_pos(self):
        return np.array([self.p1.x - self.p2.x])

    def __relative_vel(self):
        return np.array([self.p1.v - self.p2.v])

    def force_value(self):
        return self.__calculate_f_spring()

    def __calculate_f_spring(self):
        relative_pos = self.__relative_pos()
        norm_pos = np.linalg.norm(relative_pos)

        if norm_pos != 0:
            unit_vector = relative_pos / norm_pos
        else:
            unit_vector = np.array([0, 0, 0])

        f_spring = -self.__k * (norm_pos - self.__l0) * unit_vector
        return np.squeeze(f_spring)

    def calculate_jacobian(self):
        relative_pos = self.__relative_pos()
        norm_pos = np.linalg.norm(relative_pos)
        unit_vector = relative_pos / norm_pos


        # # Copied from Java code
        # Jx = np.zeros((3, 3))
        #
        # mul = self.__l0 * 1 / norm_pos - 1
        # Jx[0, 0] = unit_vector[0][0]*unit_vector[0][0] + (unit_vector[0][0]*unit_vector[0][0] - 1) * mul
        # Jx[0, 1] = unit_vector[0][0]*unit_vector[0][1] + unit_vector[0][0]*unit_vector[0][1] * mul
        # Jx[0, 2] = unit_vector[0][0]*unit_vector[0][2] + unit_vector[0][0]*unit_vector[0][2] * mul
        # Jx[1, 0] = unit_vector[0][1]*unit_vector[0][0] + unit_vector[0][1]*unit_vector[0][0] * mul
        # Jx[1, 1] = unit_vector[0][1]*unit_vector[0][1] + (unit_vector[0][1]*unit_vector[0][1] - 1) * mul
        # Jx[1, 2] = unit_vector[0][1]*unit_vector[0][2] + unit_vector[0][1]*unit_vector[0][2] * mul
        # Jx[2, 0] = unit_vector[0][2]*unit_vector[0][0] + unit_vector[0][2]*unit_vector[0][0] * mul
        # Jx[2, 1] = unit_vector[0][2]*unit_vector[0][1] + unit_vector[0][2]*unit_vector[0][1] * mul
        # Jx[2, 2] = unit_vector[0][2]*unit_vector[0][2] + (unit_vector[0][2]*unit_vector[0][2] - 1) * mul
        #
        # Jx = self.__k * Jx

        # from Python v1
        i = np.identity(3)
        T = np.matmul(np.transpose(unit_vector), unit_vector)
        jx = -self.__k * ((self.__l0 / norm_pos - 1) * (T - i) + T)       # R. Leuthold

        return jx


if __name__ == "__main__":

    particle1 = Particle([0, 0, 0], [0, 0, 0], 1, False)
    particle2 = Particle([0, 0, 1], [0, 0, 0], 1, False)
    stiffness = 1e5
    rest_length = 0
    timestep = 0.001

    springdamper = SpringDamper(particle1, particle2, stiffness, rest_length, timestep)

    print(springdamper)
    print()
    print(springdamper.force_value())
    # print((np.sqrt(3)-1)*1e5/np.sqrt(3))  # value check
    print()
    print(springdamper.calculate_jacobian())
    pass
