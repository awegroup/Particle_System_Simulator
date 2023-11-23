# -*- coding: utf-8 -*-
"""
Optical force calculation framework

Created on Tue Nov  7 14:19:21 2023

@author: Mark Kalsbeek
"""
from enum import Enum

import numpy as np
from scipy.constants import c
from src.particleSystem.Force import Force

class OpticalForceCalculator(Force):
    """
    """
    def __init__(self, ParticleSystem, LaserBeam):
        self.ParticleSystem = ParticleSystem
        self.LaserBeam = LaserBeam
        
        if not hasattr(self.ParticleSystem.particles[0],'optical_type'):
            raise AttributeError("ParticleSystem does not have any optical properties set!")
        
        
        
    def __str__(self):
        print("OpticalForceCalculator object instantiated with attributes:")
        print(f"ParticleSystem: \n {self.ParticleSystem}")
        print(f"LaserBeam: \n {self.LaserBeam}")
        return ""

    def force_value(self):
        PS = self.ParticleSystem
        LB = self.LaserBeam
        area_vectors = PS.find_surface()
        locations, _ = PS.x_v_current
        locations = np.reshape(locations, (int(len(locations)/3),3))
        
        # ! Note ! This bakes in implicitly that the orientation of the light
        # vector is in z+ direction
        intensity_vectors = np.array([[0,0,LB.intensity_profile(x,y)] for x,y,z in locations])
        
        abs_area_vectors = np.linalg.norm(area_vectors,axis=1)
        abs_intensity_vectors = intensity_vectors[:,2]
        
        cosine_factor = np.sum(area_vectors * intensity_vectors, axis=1)/ (abs_area_vectors * abs_intensity_vectors)
        
        forces = area_vectors.copy()
        
        for i in range(3):
            forces[:,i] *= abs_intensity_vectors * cosine_factor / c

        #return area_vectors, intensity_vectors, abs_area_vectors, abs_intensity_vectors, cosine_factor, forces
        return forces 
        
        
        


class ParticleOpticalPropertyType(Enum):
    """
    Enumeration representing the various types of optical properties for the Particles
    
    Attributes
    ----------
    SPECULAR : str
        Indicates that the particle reflects light specularly
    ANISOTROPICSCATTERER : str
        Indicates that the particle scatter light anisotropically
    """
    
    SPECULAR = "specular"
    ANISOTROPICSCATTERER = "anisotropicscatterer"
    
    

if __name__ == "__main__":
    from src.particleSystem.ParticleSystem import ParticleSystem
    from code_Validation.saddle_form import saddle_form
    from src.ExternalForces.LaserBeam import LaserBeam
    import matplotlib.pyplot as plt

    
    PS = saddle_form.instantiate_ps()
    #PS.stress_self()
    #for i in range(10): PS.simulate()
    for particle in PS.particles:
        particle.x[2]= 0
    
    
    I_0 = 100e9 /(10*10)
    mu_x = 5
    mu_y = 5
    sigma = 5
    LB = LaserBeam(lambda x, y: I_0 * np.exp(-1/2 *((x-mu_x)/sigma)**2
                                             -1/2 *((y-mu_y)/sigma)**2), 
                   lambda x,y: [0,1])
    LB = LaserBeam(lambda x, y: np.ones(x.shape)*I_0, lambda x,y: [0,1])
    
    for particle in PS.particles:
        particle.optical_type = ParticleOpticalPropertyType.SPECULAR
    
    ExternalForce = OpticalForceCalculator(PS, LB)
    
    forces = ExternalForce.force_value()
    
        
    ax = PS.plot()
    
    points, _ = PS.x_v_current
    points = points.reshape((int(len(points)/3),3))
    x,y,z = points[:,0], points[:,1], points[:,2]
    a_u = forces[:,0]
    a_v = forces[:,1]
    a_w = forces[:,2]   
    ax.scatter(x,y,z)
    ax.quiver(x,y,z,a_u,a_v,a_w, length = 0.1)

    #ax2 = fig.add_subplot(projection='3d')
    #LB.plot(ax2, x_range = [0,10], y_range=[0,10])

    
    
    
    
    
    
    
    
    