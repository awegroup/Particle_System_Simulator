# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:46:21 2023

@author: Mark Kalsbeek
"""
import numpy as np



class Mesher:
    def __init__(self):
        self.shapes = []
        self.points = []
        self.links = []
    
    def add_rectangle(self, 
                      upper_left, 
                      lower_right, 
                      mesh_type, 
                      mesh_edge_length=1, 
                      angle=0
                      ):
        
        self.shapes.append(Rectangle(upper_left, lower_right, mesh_type, mesh_edge_length, angle=0))
    
    def mesh_shapes(self):
        for shape in self.shapes:
            print(shape.mesh())
            points, links = shape.mesh()
            self.points.append(points)
            
            # Need to shift link indices to account for additions to point list
            indice_shift = len(points)
            for link in links:
                link[0] += indice_shift
                link[1] += indice_shift
                
            self.links.append(links)
                
class Geometry:
    def __init__(self):
        pass
    
    
    
class Rectangle(Geometry):
    def __init__(self, 
                 upper_left, 
                 lower_right, 
                 mesh_type, 
                 mesh_edge_length = 1, 
                 angle = 0):
        self.mesh_types = {'square': self.mesh_square,
                           'sq': self.mesh_square,
                           'square_cross': self.mesh_square_cross,
                           'sqx': self.mesh_square_cross}
        self.upper_left = upper_left
        self.lower_right = lower_right
        self.mesh_type = mesh_type
        self.angle = angle
        
        
    def mesh(self):
        return self.mesh_types[self.mesh_type]()

    
    def mesh_square(self, pattern_angle = 0):
        return [1,0], [[1,0]]
    
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


if __name__ == '__main__':
    from src.particleSystem.Particle import Particle
    from src.particleSystem.SpringDamper import SpringDamper 
    from src.particleSystem.ParticleSystem import ParticleSystem
    
    mesh = Mesher()
    mesh.add_rectangle(0, 0, 'square')
    mesh.mesh_shapes()
    
    testrect = Rectangle(0,0,'square')