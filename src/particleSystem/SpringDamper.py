"""
Child Class 'SpringDamper', for spring-damper objects to be instantiated in ParticleSystem
""" 
from enum import Enum
import numpy as np

from src.particleSystem.ImplicitForce import ImplicitForce
from src.particleSystem.Particle import Particle


class SpringDamperType(Enum):
    """
    Enumeration representing the various types of SpringDamper objects.
    
    Attributes
    ----------
    DEFAULT : str
        Represents the default SpringDamper type, which has standard characteristics.
    NONCOMPRESSIVE : str
        Represents a SpringDamper that cannot be compressed, only stretched.
    NONTENSILE : str
        Represents a SpringDamper that cannot be stretched, only compressed.
    
    Notes
    -----
    The SpringDamper type affects the initialization and behavior of the SpringDamper objects.
    Each type might have specific properties or behaviors associated with it in the SpringDamper class.
    """
    DEFAULT = "default"
    NONCOMPRESSIVE = "noncompressive"
    NONTENSILE = "nontensile"

class SpringDamper(ImplicitForce):
    """
    #TODO one line summary
    
    
    Attributes: #TODO finish this
        
    """
    def __init__(self, p1: Particle, p2: Particle, k: float, c: float, linktype=SpringDamperType.DEFAULT):
        """Initializes the spring damper
        
        Args:
            p1, p2: 
                The two Particle instances to be connected
            k: 
                A float representing the stiffness of the spring in N/m.
            c: 
                A float representing the damping coefficient in Ns/m
            linktype:
                A SpringDamperType enum representing the properties of the link
                See SpringDamperType for more information
                
        """
        super().__init__(p1, p2)
        self.__k = k
        self.__c = c
        self.__l0 = np.linalg.norm(self.__relative_pos())
        self.__linktype = linktype
        return

    def __str__(self):
        return f"SpringDamper object, spring stiffness [n/m]: {self.__k}, rest length [m]: {self.l0}\n" \
               f"Damping coefficient [N s/m]: {self.__c}\n" \
               f"Assigned particles\n  p1: {self.p1}\n  p2: {self.p2}\n"\
               f"Link type: {self.__linktype}"

    def __relative_pos(self):
        return np.array([self.p1.x - self.p2.x])

    def __relative_vel(self):
        return np.array([self.p1.v - self.p2.v])

    def force_value(self):
        if self.__linktype == SpringDamperType.DEFAULT:
            return self.__calculate_f_spring() + self.__calculate_f_damping()
        
        elif self.__linktype == SpringDamperType.NONCOMPRESSIVE:
            l = np.linalg.norm(self.__relative_pos())
            if l >=self.l0:
                return self.__calculate_f_spring() + self.__calculate_f_damping()
            else:
                return np.array([0, 0, 0])
            
        elif self.__linktype == SpringDamperType.NONTENSILE:
            l = np.linalg.norm(self.__relative_pos())
            if l <=self.l0:
                return self.__calculate_f_spring() + self.__calculate_f_damping()
            else:
                return np.array([0, 0, 0])

    def __calculate_f_spring(self):
        relative_pos = self.__relative_pos()
        norm_pos = np.linalg.norm(relative_pos)

        if norm_pos != 0:
            unit_vector = relative_pos / norm_pos
        else:
            unit_vector = np.array([0, 0, 0])

        f_spring = -self.__k * (norm_pos - self.l0) * unit_vector
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

        # Using guard classes to return early in special cases
        if (
                self.__linktype == SpringDamperType.NONCOMPRESSIVE and
                norm_pos <= self.__l0
            ):
            return np.zeros((3, 3)), np.zeros((3, 3))
        
        elif (
                self.__linktype == SpringDamperType.NONTENSILE and
                norm_pos >= self.__l0
            ):
            return np.zeros(3), np.zeros(3)

        if norm_pos != 0:
            unit_vector = relative_pos / norm_pos
        else:
            norm_pos = 1
            unit_vector = np.array([0, 0, 0])

        i = np.identity(3)
        T = np.matmul(np.transpose(unit_vector), unit_vector)
        jx = -self.__k * ((self.l0 / norm_pos - 1) * (T - i) + T)

        jv = -self.__c*i

        return jx, jv
    
    def line_segment(self):
        """Returns coordinate tuple of particles at either end of segment"""
        return (self.p1.x, self.p2.x)
    
    @property
    def l0(self):
        return self.__l0
    
    @l0.setter
    def l0(self,value): # Exposed to enable self-stressing of mesh
        self.__l0 = value

    @property
    def c(self):
        return self.__c
    
    @c.setter
    def c(self,value): # Exposed to enable resetting when using kinetic damping
        self.__c = value


if __name__ == "__main__":

    particle1 = Particle([0, 0, 0], [0, 0, 0], 1, False)
    particle2 = Particle([0, 0, 1], [0, 0, 1], 1, False)
    stiffness = 1e5
    damping = 10
    rest_length = 0
    linktype = 'noncomp'

    springdamper = SpringDamper(particle1, particle2, stiffness, damping)

    print(springdamper)
    print()
    print(springdamper.force_value())
    # print((np.sqrt(3)-1)*1e5/np.sqrt(3))  # value check
    print()
    print(springdamper.calculate_jacobian())
    pass
