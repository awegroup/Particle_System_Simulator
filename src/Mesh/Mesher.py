# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:46:21 2023

@author: Mark Kalsbeek
"""
import numpy as np

from src.particleSystem.Particle import Particle
from src.particleSystem.SpringDamper import SpringDamper 
from src.particleSystem.ParticleSystem import ParticleSystem


class Mesher:
    def __init__(self):
        self.shapes = []
    
    def add_rectangle(self, upper_left, lower_right, mesh_type):
        self.shapes.append(Rectangle(upper_left, lower_right, mesh_type))
    
    def mesh_shapes(self):
        return


class Geometry:
    def __init__(self):
        pass
    
    
    
class Rectangle(Geometry):
    def __init__(self, upper_left, lower_right, angle = 0):
        self.upper_left = upper_left
        self.lower_right = lower_right
        self.angle = angle
        
    
    def mesh_square(self, pattern_angle = 0):
        return
    
    def mesh_triangular(self, pattern_angle = 0):
        return
    
    def mesh_hex(self, pattern_angle = 0):
        return
    
    def mesh_square_cross(self, pattern_angle = 0):
        return
    
    
class Ellipse(Geometry):
    def __init__(self, center, radius, eccentricity = 0, angle = 0):
        self.center = center
        self.radius = radius
        self.eccentricity = eccentricity
        self.angle = angle
    
    
    def mesh_square(self):
        return
    
    def mesh_triangular(self):
        return
    
    def mesh_hex(self):
        return
    
    def mesh_square_cross(self):
        return