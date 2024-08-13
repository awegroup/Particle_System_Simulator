"""
Child Abstract Base Class 'Force', for force objects to be instantiated in ParticleSystem
"""

from Particle_System_Simulator.particleSystem.SystemObject import SystemObject
from abc import abstractmethod


class Force(SystemObject):

    def __init__(self):
        super().__init__()
        return

    def __str__(self):
        return

    @abstractmethod
    def force_value(self):
        return


if __name__ == "__main__":
    pass
