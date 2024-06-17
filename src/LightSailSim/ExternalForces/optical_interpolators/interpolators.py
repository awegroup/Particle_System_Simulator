# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:01:37 2024

@author: Mark Kalsbeek
"""
from typing import Callable
import logging
from functools import lru_cache
import os.path

import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from LightSailSim.particleSystem.ParticleSystem import ParticleSystem
from LightSailSim.ExternalForces.OpticalForceCalculator import wrap_spherical_coordinates

# Setup path for abs. file imports
my_path = os.path.abspath(os.path.dirname(__file__))

PhC_library = {
        'dummy':'dummy.csv',
        'Gao':'PhC_Gao_et_al.csv',
        'Mark_2':'Mark_2_export.csv',
        'Mark_3':'Mark_3_export.csv',
        'Mark_4':'Mark_4_export.csv',
        'Mark_4.1':'Mark_4.1_export.csv',
        'Mark_5':'Mark_5_export.csv',
        'Mark_6':'Mark_6.csv',
        'Mark_7':'Mark_7_export.csv',
        'Mark_8':'Mark_8_export.csv',
        'Mark_9':'Mark_9_export.csv'
    }

def create_interpolator_specular() -> Callable:
    polar_in = np.linspace(0,np.pi,10)
    azimuth_in = np.linspace(0,2*np.pi,10)
    polarization_in = np.linspace(0,np.pi/2,10)
    polar_in, azimuth_in, polarization_in = np.meshgrid(polar_in, azimuth_in, polarization_in)

    polar_in = polar_in.reshape([1000,1])
    azimuth_in = azimuth_in.reshape([1000,1])
    polarization_in = polarization_in.reshape([1000,1])

    incidence = np.hstack((polar_in, azimuth_in, polarization_in))

    polar_out =  polar_in
    azimuth_out = np.pi + azimuth_in
    azimuth_out = azimuth_out%(2*np.pi)
    magnitude = np.ones(azimuth_out.shape)
    out = np.hstack((polar_out, azimuth_out, magnitude))

    optical_interpolator = LinearNDInterpolator(incidence,out)

    return optical_interpolator

def create_interpolator(fname: str, rotation:float = 0)-> Callable:
    """
    create interpolator from simulation data

    Parameters
    ----------
    fname : string
        Path to the data.
    rotation : float
        Rotation around z+ axis of photonic crystal. Allows to represent crystal in different
        oriantations. [rad]

    Returns
    -------
    Callable
        Interpolator for optical behaviour.

    """
    path = os.path.join(my_path, fname)
    data = np.loadtxt(path, delimiter = ',',comments='#')
    incidence = data[:,:3]
    out = data[:,3:]

    if not np.any(np.pi*2 in incidence[:,1]):
        mask = incidence[:,1] == 0
        in_dupes = incidence.copy()[mask]
        out_dupes = out.copy()[mask]
        in_dupes[:,1] += 2*np.pi

        incidence = np.vstack((incidence,in_dupes))
        out = np.vstack((out,out_dupes))

    return linear_interpolator(incidence,out,rotation, name=fname)

class linear_interpolator():
    """
    maps [theta, phi, pol] to [theta, phi, mag]

    cache_values : bool
        enables lru caching for call function.  Note, you only want to use caching if you are
        feeding coordinate tuples. Breaks when numpy arrays are fed in!
    """
    def __init__(self, coordinates, values, rotation, cache_values = False, name=None):
        self.coordinates = coordinates
        self.cache_values = cache_values
        self.values = values
        self.tree = KDTree(coordinates)
        self.rotation = rotation
        self.interp = LinearNDInterpolator(coordinates, values)

        if self.cache_values:
            self.__call__ =  lru_cache(maxsize=None)(self.__call__)


    def __call__(self,coordinates):
        #coordinates = coordinates.copy()-np.array([0,self.rotation,self.rotation])
        coordinates = coordinates-np.array([0,self.rotation,self.rotation])
        if len(coordinates.shape)>1:
            coordinates = np.round(coordinates, 8)
            coordinates[:,1]%=2*np.pi
            pol = coordinates[:,2]
            x = np.abs(np.cos(pol))
            y = np.abs(np.sin(pol))
            coordinates[:,2] = np.arctan(y/x)

            v = self.interp(coordinates)
            v[:,1]+=self.rotation
            v[:,1]%=2*np.pi
            if np.any(np.isnan(v)):
                for i, line in enumerate(v):
                    if np.any(np.isnan(line)):
                        logging.warning("Interpolation error resulting in nan values for input "+str(coordinates[i,:]))
            v = np.nan_to_num(v)

        else:
            coordinates[1]%=2*np.pi
            pol = coordinates[2]
            x = abs(np.cos(pol))
            y = abs(np.sin(pol))
            coordinates[2] = np.arctan(y/x)

            v = self.interp(coordinates)[0]
            v[1]+=self.rotation
            v[1]%=2*np.pi
            if any(np.isnan(v)):
                logging.warning("Interpolation error resulting in nan values for input "+str(coordinates))
        return v

    def old__call__(self, coordinates):
        coordinates = coordinates.copy()-np.array([0,self.rotation,self.rotation])
        coordinates[1]%=2*np.pi
        pol = coordinates[2]
        x = abs(np.cos(pol))
        y = abs(np.sin(pol))
        coordinates[2] = np.arctan(y/x)

        dist, ind = self.tree.query(coordinates, k=2)
        d1, d2 = dist.T
        v1,v2 = self.values[ind]

        if d1 == 0:
            return v1
        elif d2 ==0:
            return v2
        else:
            v = (d1)/(d1 + d2)*(v2 - v1) + v1
            #print(self.rotation, coordinates, v)
            v[1]+=self.rotation
            v[1]%=2*np.pi
            return v




def check_interpolator(interp, coordinates, ax = None):
    theta, phi, pol = coordinates
    theta_out, phi_out, pol_out = wrap_spherical_coordinates(*[np.array(i,dtype=float) for i in coordinates])
    theta_out,phi_out,mag_out = interp([theta_out, phi_out, pol_out])


    in_x = np.sin(theta) * np.cos(phi)
    in_y = np.sin(theta) * np.sin(phi)
    in_z = np.cos(theta)

    out_x = np.sin(theta_out) * np.cos(phi_out) * mag_out
    out_y = np.sin(theta_out) * np.sin(phi_out) * mag_out
    out_z = np.cos(theta_out) * mag_out

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    ax.set_zlim((-1,1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')

    ax.quiver(0,0,0,-in_x,-in_y,-in_z, label='incident ray', color='y',pivot='tip')
    ax.quiver(0,0,0,out_x,out_y,out_z, label='net scattered ray', color='r',pivot='tail')
    ax.legend()



# PhC_specular = create_interpolator_specular()
# PhC_Gao = create_interpolator(crystal_dict['Gao'])
if __name__ == '__main__':
    dummy = create_interpolator(PhC_library['dummy'], 0)
    dummy = create_interpolator(PhC_library['Mark_4.1'], 0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for theta in np.linspace(np.deg2rad(-15),np.deg2rad(15), 5):
        vec = [theta,np.pi,0]
        check_interpolator(dummy,vec,ax)

        print(vec, "\t", dummy(vec))
        print(wrap_spherical_coordinates(*[np.array(i,dtype=float) for i in vec]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('Demonstrate sampled points')
    ax.set_xlim((-0.5,0.5))
    ax.set_ylim((-0.5,0.5))
    ax.set_zlim((-1,1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')

    mask = np.isclose(dummy.coordinates[:,2],np.pi/2)
    out = dummy.values[mask]
    incidence = dummy.coordinates[mask]
    theta= incidence[:,0]
    phi = incidence[:,1]

    n_x = np.sin(theta) * np.cos(phi)
    n_y = np.sin(theta) * np.sin(phi)
    n_z = np.cos(theta)

    theta_out = out[:,0]
    phi_out = out[:,1]
    mag_out = out[:,2]

    s_x = np.sin(theta_out) * np.cos(phi_out) * mag_out
    s_y = np.sin(theta_out) * np.sin(phi_out) * mag_out
    s_z = np.cos(theta_out) * mag_out

    ax.quiver(0,0,0,-n_x,-n_y,-n_z, label='incident ray', color='y',pivot='tip')
    ax.quiver(0,0,0,s_x, s_y, s_z, label='net scattered ray', color='r',pivot='tail')
    ax.legend()