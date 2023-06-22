"""
Child Class 'SpringDamper', for spring-damper objects to be instantiated in ParticleSystem
"""
from Msc_Alexander_Batchelor.src.particleSystem.ImplicitForce import ImplicitForce
from Msc_Alexander_Batchelor.src.particleSystem.Particle import Particle
import numpy as np


class SpringDamper(ImplicitForce):

    def __init__(self, p1: Particle, p2: Particle, k: float, l0: float, c: float, dt: float):
        self.__k = k
        self.__c = c
        self.__l0 = l0
        self.__dt = dt
        super().__init__(p1, p2)
        return

    def __str__(self):
        return f"SpringDamper object, spring stiffness [n/m]: {self.__k}, rest length [m]: {self.__l0}\n" \
               f"Damping coefficient [N s/m]: {self.__c}" \
               f"Assigned particles\n  p1: {self.p1}\n  p2: {self.p2}"

    def __relative_pos(self):
        return np.array([self.p1.x - self.p2.x])

    def __relative_vel(self):
        return np.array([self.p1.v - self.p2.v])

    def force_value(self):
        return self.__calculate_f_spring(), self.__calculate_f_damping()

    def __calculate_f_spring(self):
        relative_pos = self.__relative_pos()
        norm_pos = np.linalg.norm(relative_pos)

        if norm_pos != 0:
            unit_vector = relative_pos / norm_pos
        else:
            unit_vector = np.array([0, 0, 0])

        f_spring = -self.__k * (norm_pos - self.__l0) * unit_vector
        return np.squeeze(f_spring)

    def __calculate_f_damping(self):
        relative_pos = self.__relative_pos()
        relative_vel = np.squeeze(self.__relative_vel())
        norm_pos = np.linalg.norm(relative_pos)

        if norm_pos != 0:
            unit_vector = np.squeeze(relative_pos / norm_pos)
        else:
            unit_vector = np.squeeze(np.array([0, 0, 0]))

        f_damping = -self.__c * np.dot(relative_vel, unit_vector) * unit_vector
        return np.squeeze(f_damping)

    def calculate_jacobian(self):
        relative_pos = self.__relative_pos()
        norm_pos = np.linalg.norm(relative_pos)

        if norm_pos != 0:
            unit_vector = relative_pos / norm_pos
        else:
            norm_pos = 1
            unit_vector = np.array([0, 0, 0])

        i = np.identity(3)
        T = np.matmul(np.transpose(unit_vector), unit_vector)
        jx = -self.__k * ((self.__l0 / norm_pos - 1) * (T - i) + T)

        jv = -self.__c*i

        return jx, jv


if __name__ == "__main__":

    particle1 = Particle([0, 0, 0], [0, 0, 0], 1, False)
    particle2 = Particle([0, 0, 1], [0, 0, 1], 1, False)
    stiffness = 1e5
    damping = 10
    rest_length = 0
    timestep = 0.001

    springdamper = SpringDamper(particle1, particle2, stiffness, rest_length, damping, timestep)

    print(springdamper)
    print()
    print(springdamper.force_value())
    # print((np.sqrt(3)-1)*1e5/np.sqrt(3))  # value check
    print()
    print(springdamper.calculate_jacobian())
    pass
