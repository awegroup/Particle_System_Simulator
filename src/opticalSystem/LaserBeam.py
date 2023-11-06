# -*- coding: utf-8 -*-
"""
Child Class 'LaserBeam', for holding laser atributes
"""
from typing import Callable

import numpy as np
from src.particleSystem.SystemObject import SystemObject

class LaserBeam(SystemObject):
    """
    Holds information about the laserbeam
    
    Attributes
    ---------
    intensity_profile : Callable[[float, float], float]
        Maps (x, y) to the scalar intensity profile of the beam.
    polarization_map : Callable[[float, float], np.ndarray]
        Maps (x, y) to the polarization profile of the beam represented as a Jones vector.
        
    """    
    def __init__(self, 
                 intensity_profile: Callable[[float, float], float],
                 polarization_map: Callable[[float, float], list[np.complex_, np.complex_]]
                 ):
        """
        Initializes a laserbeam based on input parameters
        
        The polarization profile represents the Jones vector. Its datatype is 
        allowed to be complex in order to capture both linear and circular 
        polarization states. 
        

        Parameters
        ----------
        intensity_profile : Callable[[float, float], float]
            A numpy compatible function that maps x, y to the scalar intensity profile of the beam.
        polarization_map : Callable[[float, float], np.ndarray]
            A numpy compatible function that maps x, y to the polarization profile of the beam.
        Returns
        -------
        None.

        """
        self.intensity_profile = intensity_profile
        self.polarization_map = polarization_map
    
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mu = 0
    sigma = 0.5
    LB = LaserBeam(lambda x, y: np.exp(-1/2 *((x-mu)/sigma)**2
                                       -1/2 *((y-mu)/sigma)**2),
                   lambda x, y: np.array([1+0j, 0+0j])
                   )
    x = np.linspace(-1,1,10)
    y = np.linspace(-1,1,10)
    x, y = np.meshgrid(x,y)
    
    ip = LB.intensity_profile(x,y)
    pol = LB.polarization_map(x,y)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(x,y,ip, label = "Intensity")
    ax.quiver(x,y,ip, pol[0],pol[1],0, length = 0.1, color='r', label = "Polarization")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.legend()
    