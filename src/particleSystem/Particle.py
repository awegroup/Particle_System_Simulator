"""
Child Class 'Particle', for particle objects to be instantiated in ParticleSystem
"""
from src.particleSystem.SystemObject import SystemObject
import numpy as np
import numpy.typing as npt


class Particle(SystemObject):
    def __init__(self, 
                 x: npt.ArrayLike, 
                 v: npt.ArrayLike, 
                 m: float, 
                 fixed: bool, 
                 constraint: npt.NDArray = None,
                 constraint_type: str = 'free'):
        """
        Object that holds particle data

        Parameters
        ----------
        x : npt.ArrayLike
            Position (x,y,z) in meter
        v : npt.ArrayLike
            Velocity (x,y,z) in meters per second
        m : float
            Mass in kilograms
        fixed : bool
            Wether or not the particle is fixed
        constraint : npt.NDArray, optional
            Desrcribes specific constraint if particle is fixed. The default is 
            None. This indicates that it's fixed in all three dimentions.
        constraint_type : str, optional
            Describes constraint type. Can be free, point, line or plane
            Can be left default for fixed points, as they're indicated by
            passing constraint = [0,0,0]
            
        Raises
        ------
        AttributeError
            Raises error if constraint is set incorrectly.

        Returns
        -------
        None.

        """
        self.__x = np.array(x, dtype='float64')
        self.__v = np.array(v, dtype='float64')
        self.__m = m
        self.__fixed = fixed
        self.__constraint = None
        self.__constraint_type = constraint_type.lower()
        

        if self.__fixed:
            self.validate_constraint(constraint)
            self.constraint_projection()
        super().__init__()


    def __str__(self):
        return (f"Particle Object, position [m]: [{self.__x[0]}, {self.__x[1]}, {self.__x[2]}], " 
               f"velocity [m/s]: [{self.__v[0]}, {self.__v[1]}, {self.__v[2]}], mass [kg]: {self.__m}" 
               f", fixed: {self.__fixed}, {self.__constraint=}, {self.__constraint_type=}")

    def validate_constraint(self, constraint):
        "Checks if constraint is entered correctly, raises exception if otherwise"
        if self.__fixed:
            if constraint == None:
                constraint = [0,0,0]
                self.__constraint_type = 'point'
            if self.__constraint_type not in ['point', 'line', 'plane']:
                raise AttributeError(f"Incorrect constraint type set, expected"
                                     f" line or plane, got "
                                     f"{self.__constraint_type}")
            try: 
                self.__constraint = np.array(constraint, dtype=float).reshape(1, 3)
            except (ValueError, TypeError) as e:
                raise AttributeError(f"Particle set as 'fixed' but constraint "
                                     f"not set correctly. Expecting (1,3) "
                                     f"npt.Arraylike, instead got "
                                     f"{constraint=}. Error: {e}")
        else:
            self.__constraint = None

    def constraint_projection(self):
        if np.sum(self.__constraint == 0) == 3:
            self.constraint_projection_matrix = np.zeros((3,3))
        else:
            normalised_constraint = self.__constraint / np.linalg.norm(self.__constraint)
            if self.__constraint_type == 'plane':
                projection_matrix = np.eye(3) - np.outer(normalised_constraint, 
                                                         normalised_constraint)
                self.constraint_projection_matrix = projection_matrix
            elif self.__constraint_type == 'line':
                projection_matrix = np.outer(normalised_constraint, 
                                             normalised_constraint)
                self.constraint_projection_matrix = projection_matrix
                
    def update_pos(self, new_pos: npt.ArrayLike):
        if not self.__fixed:
            self.__x = np.array(new_pos)
        else:
            self.__x += self.constraint_projection_matrix.dot(np.array(new_pos) - self.__x)

    
    def update_pos_unsafe(self, new_pos : npt.ArrayLike):
        """position update method that will override locations of fixed nodes"""
        self.__x = np.array(new_pos)

    def update_vel(self, new_vel: npt.ArrayLike):
        if not self.__fixed:
            self.__v = np.array(new_vel)
        else:
            self.__v += self.constraint_projection_matrix.dot(np.array(new_vel) - self.__v)


    @property
    def x(self):
        return self.__x

    @property
    def v(self):
        return self.__v

    @property
    def m(self):
        return self.__m

    @property
    def fixed(self):
        return self.__fixed
    
    def set_fixed(self, fixed, constraint = None, constraint_type = 'free'):
        self.__fixed = fixed
        self.validate_constraint(constraint)
        self.__constraint_type = constraint_type
        if self.__fixed:
            self.constraint_projection()
        


if __name__ == "__main__":
    position = [0, 0, 0]
    velocity = [0, 0, 0]
    mass = 1
    fixed1 = False
    fixed2 = True
    constraint2 = [-1,0,1]
    constraint_type2 = 'line'
    p1 = Particle(position, velocity, mass, fixed1)
    p2 = Particle(position, velocity, mass, fixed2, constraint2, constraint_type2)
    print('Starting positions')
    print(p1)
    print(p2, '\n')
    updated_pos = [0, 1, 1]
    updated_vel = [0, 0, 1]
    p1.update_pos(updated_pos)
    p1.update_vel(updated_vel)
    p2.update_pos(updated_pos)
    p2.update_vel(updated_vel)
    print('Updated positions')
    print(p1)
    print(p2)
