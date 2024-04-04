# -*- coding: utf-8 -*-
"""
Optical force calculation framework

Created on Tue Nov  7 14:19:21 2023

@author: Mark Kalsbeek
"""
from enum import Enum
from itertools import compress


import numpy as np
import numpy.typing as npt
import scipy as sp
from scipy.constants import c
from scipy.spatial.transform import Rotation
from src.particleSystem.Force import Force
import logging

class OpticalForceCalculator(Force):
    """
    Handles the calculation of forces arising from optical pressure
    """
    def __init__(self, ParticleSystem, LaserBeam):
        self.ParticleSystem = ParticleSystem
        self.PS = self.ParticleSystem #alias for convenience
        self.LaserBeam = LaserBeam

        if not hasattr(self.ParticleSystem.particles[0],'optical_type'):
            raise AttributeError("ParticleSystem does not have any optical properties set!")

        super().__init__()
        return


    def __str__(self):
        print("OpticalForceCalculator object instantiated with attributes:")
        print(f"ParticleSystem: \n {self.ParticleSystem}")
        print(f"LaserBeam: \n {self.LaserBeam}")
        return ""

    def force_value(self):
        """
        Calculates optical forces based on optical properties of ParticleSystem and LaserBeam

        Returns
        -------
        forces : npt.NDArray
            flattened array of external forces of length 3 * n_particles.

        """
        PS = self.ParticleSystem
        LB = self.LaserBeam
        area_vectors = PS.find_surface()
        locations, _ = PS.x_v_current_3D
        forces = np.zeros(locations.shape)

        if not hasattr(self, 'optical_type_mask'):
            self.create_optical_type_mask()

        # ! Note ! This bakes in implicitly that the orientation of the light
        # vector is in z+ direction
        intensity_vectors = np.array([[0,0,LB.intensity_profile(x,y)] for x,y,z in locations])
        polarisation_vectors = LB.polarization_map(locations[:,0],locations[:,1])

        for optical_type in self.optical_type_mask.keys():
            if optical_type == ParticleOpticalPropertyType.SPECULAR:
                mask = self.optical_type_mask[optical_type]
                forces[mask] = self.calculate_specular_force(area_vectors[mask],
                                                             intensity_vectors[mask])

            elif optical_type == ParticleOpticalPropertyType.AXICONGRATING:
                mask = self.optical_type_mask[optical_type]
                filtered_particles = compress(PS.particles, mask)
                axicon_angle = [p.axicon_angle for p in filtered_particles]
                forces[mask] = self.calculate_axicongrating_force(area_vectors[mask],
                                                                  intensity_vectors[mask],
                                                                  axicon_angle)

            elif optical_type == ParticleOpticalPropertyType.ARBITRARY_PHC:
                mask = self.optical_type_mask[optical_type]
                filtered_particles = compress(PS.particles, mask)

                # consider splitting this into a  "create_optical_property_list" function
                # because now we're doing these loops every time you call force_value
                optical_interpolators = [p.optical_interpolator for p in filtered_particles]

                forces[mask] = self.calculate_arbitrary_phc_force(area_vectors[mask],
                                                                  intensity_vectors[mask],
                                                                  polarisation_vectors[mask],
                                                                  optical_interpolators)
        return forces

    def calculate_specular_force(self, area_vectors, intensity_vectors):
        """
        Calculates forces for particles of optical type 'specular'

        !!! TODO implement reflectivity coefficient

        Parameters
        ----------
        area_vectors : npt.NDArray
            n_particles x 3 array of area vectors
        intensity_vectors : npt.NDArray
            n_particles x 3 array of laser beam intensity vectors

        Returns
        -------
        forces : npt.NDArray
            flattened array of external forces of length 3 * n_particles.
        """
        # First we compute the incident power on the particle areas
        abs_area_vectors = area_vectors[:,2] # assumes z+ poynting vector
        abs_intensity_vectors = intensity_vectors[:,2] # assumes z+ poynting vector
        incident_power = abs_area_vectors * abs_intensity_vectors

        # To get the direction of the forces we need to normalise the area
        # vectors. For convenience we roll that into the force calculation of
        # dF = dP/c. We double it because the PhC is acting in reflection
        norms = np.linalg.norm(area_vectors, axis=1)
        forces = area_vectors.copy()
        for i in range(3):
            forces[:,i] *= 2*incident_power / (c*norms)

        return forces

    def calculate_axicongrating_force(self,
                                      area_vectors,
                                      intensity_vectors,
                                      axicon_angle):
        """
        Calculates forces for particles of optical type 'axicon grating'

        Parameters
        ----------
        area_vectors : npt.NDArray
            n_particles x 3 array of area vectors
        intensity_vectors : npt.NDArray
            n_particles x 3 array of laser beam intensity vectors
        axicon_angle : npt.NDArray
            3 x 3 array representing a rotation of the surface normal vector
            this determines the directions of the resulting optical forces


        Returns
        -------
        forces : npt.NDArray
            flattened array of external forces of length 3 * n_particles.
        """
        rotation_super_matrix = sp.linalg.block_diag(*axicon_angle)

        forces = self.calculate_specular_force(area_vectors, intensity_vectors)
        forces = rotation_super_matrix.dot(np.hstack(forces).T)
        forces = np.reshape(forces, [int(forces.shape[0]/3),3])

        # The forces need to be scaled to account for the fact that
        # |[1,1]| != |[1]|+|[1]|
        # We don't have acces to the angle, but we can make use of the cosine
        # rule: cos(alpha) = A.dot(B) / (|A| |B|) to get the angle between
        # z+ and the line of action of the force.
        unit_z = np.array([0,0,1])
        scaling_factor = np.matmul(axicon_angle, unit_z).dot(unit_z)

        for i in range(3):
            forces[:,i]*=scaling_factor

        return forces

    def calculate_arbitrary_phc_force(self,
                                              area_vectors,
                                              intensity_vectors,
                                              polarisation_vectors,
                                              optical_interpolators):
        """
        Calculates forces for particles of optical type 'axicon grating'

        Parameters
        ----------
        area_vectors : npt.NDArray
            An array of shape (n_particles, 3) representing the area vectors of n particles.
        intensity_vectors : npt.NDArray
            An array of shape (n_particles, 3) representing the intensity vectors of laser beams for
            n particles.
        polarisation_vectors : npt.NDArray
            An array of shape (n_particles, 3) representing the polarisation vectors of laser beams
            for n particles.


        Returns
        -------
        forces : npt.NDArray
            flattened array of external forces of length 3 * n_particles.
        """
        # Convert area vector to spherical coordinates
        result = compute_spherical_coordinates(area_vectors, polarisation_vectors)
        polar_angles, azimuth_angles, polarisation_angles = result

        # Switch reference frame; made easier becasue we a coming from [0,0,1]
        # azimuth_angles += np.pi
        # azimuth_angles %= 2*np.pi


        # Condition them for the interpolator:
        wrapped_coordinates = wrap_spherical_coordinates(polar_angles,
                                                         azimuth_angles,
                                                         polarisation_angles)
        polar_angles, azimuth_angles, polarisation_angles = wrapped_coordinates

        incoming_ray = np.vstack((polar_angles,
                                  azimuth_angles,
                                  polarisation_angles)).T

        # Find directions of outgoing rays
        # Interpolator([polar_in, azimuth_in, polarization_in])->[polar_out, azimuth_out, magnitude]
        reflected_ray = [interp(incoming_ray[i])
                         for i, interp
                         in enumerate(optical_interpolators)] # [polar_out, azimuth_out, magnitude]
        polar_angles_out, azimuth_angles_out, magnitudes = np.array(reflected_ray).T

        # Switch reference frame again; made easier becasue we are going to [0,0,1]
        polar_angles_out += polar_angles

        reflected_vectors = spherical_to_cartesian(polar_angles_out,
                                                   azimuth_angles_out,
                                                   magnitudes)

        # Compute the incident power on the particle areas
        abs_area_vectors = area_vectors[:,2] # assumes z+ poynting vector
        abs_intensity_vectors = intensity_vectors[:,2] # assumes z+ poynting vector
        incident_power = abs_area_vectors * abs_intensity_vectors

        scattered_power = reflected_vectors*incident_power[:,np.newaxis]

        net_power = np.hstack((np.zeros((incident_power.shape[0],2)),incident_power[:,np.newaxis])) + scattered_power
        forces = net_power/c

        return forces

    def create_optical_type_mask(self):
        """
        loops over particles and sets a dict of masks onto self formatted as {type:mask}

        This is used to efficiently split computation of the different particle
        types without resorting to repeated looping.

        Raises
        ------
        AttributeError
            Raises error when particles have no optical type set.

        """
        optical_type_list = []
        error_index_list = []
        for i, particle in enumerate(self.ParticleSystem.particles):
            if hasattr(particle, 'optical_type'):
                optical_type_list.append(particle.optical_type)
            else:
                error_index_list.append(i)
        if len(error_index_list)>0:
            raise AttributeError("All particles should have an optical type"
                                 " set prior to calculation of optical forces."
                                 " Currently the particles with indices"
                                 f" {error_index_list} have no property set")
        optical_type_list = np.array(optical_type_list)

        self.optical_type_mask = {}

        for optical_type in ParticleOpticalPropertyType:
            mask = optical_type_list == optical_type
            if sum(mask)>0:
                self.optical_type_mask[optical_type] = mask


    def calculate_stability_coefficients(self, displacement_range = [0.1, 5]):
        """
        Calculates the stability coefficients for the particle system

        Arguments
        ---------
        displacement_range : list
            list of length two representing the displacement magnitudes to
            perform the stability test. First value represents lateral
            displacement in meters. Second value represents
            tilt angle around the centre of mass in degrees.

        Returns
        -------
        stability_matrix : npt.arraytype
            6x6 matrix holding the stability terms of the system using
            notation convention of Jacobian.
            Unit of first three N/m, next three N/deg

        """
        q, alpha = displacement_range
        displacement_vectors = np.array([[q,0,0,0,0,0],
                                         [0,q,0,0,0,0],
                                         [0,0,q,0,0,0],
                                         [0,0,0,alpha,0,0],
                                         [0,0,0,0,alpha,0],
                                         [0,0,0,0,0,alpha]])

        jacobian = np.zeros((6,6))
        for i, vector in enumerate(displacement_vectors):
            jacobian[:,i] =np.hstack(self.calculate_force_gradient(vector))

        return jacobian


    def calculate_force_gradient(self, displacement_vector : npt.ArrayLike):
        """
        Calculates force and moment coefficients of ParticleSystems based on a 1 DOF displacement

        Parameters
        ----------
        displacement_vector : npt.ArrayLike
            1x6 vector ([x,y,z,rx,ry,rz]) representing the displacement. All but one should be equal to zero

        Raises
        ------
        AttributeError
            Raises error if multiple displacements are supplied.

        Returns
        -------
        k_trans : list
            lenght 3 list of translational reaction coefficients [dF_x/dx__i, dF_y/dx__i, dF_z/dx__i]
        k_rot : TYPE
            lenght 3 list of translational reaction coefficients [dM_x/dx__i, dM_y/dx__i, dM_z/dx__i]
        """
        displacement =  displacement_vector[displacement_vector !=0]
        if len(displacement)>1:
            raise AttributeError("Expected vector with only one nonzero value,"
                                 f"instead got {displacement_vector}")

        original = self.calculate_restoring_forces()
        self.displace_particle_system(displacement_vector)
        reaction = self.calculate_restoring_forces()
        self.un_displace_particle_system()

        k_trans = (reaction[0] - original[0])/displacement
        k_rot = (reaction[1] - original[1])/displacement
        return k_trans, k_rot

    def displace_particle_system(self, displacement : list):
        """
        displaces the associated particle system with the prescribed amount
        around the center of mass.

        Arguments
        ----------
        displacement_range : list
            list of length 6 representing the displacement magnitudes to
            perform the stability test. First three values represent lateral
            displacement in meters. Next three values represent
            tilt angle around the centre of mass in degrees.
        """
        PS = self.ParticleSystem
        if len(displacement) != 6:
            raise AttributeError("Expected list of 6 arguments representing "
                                 "x,y,z,rx,ry,rz, got list of length {} instead".format(len(displacement)))
        if hasattr(self.ParticleSystem, 'current_displacement'): #
            if (type(self.ParticleSystem.current_displacement) != type(None)
                and not
                np.all(self.ParticleSystem.current_displacement == -np.array(displacement))):
                # I want to allow this behavior, but also inform user that by doing it this way they're breaking stuff
                logging.warning(f"Particle system is already displaced: {self.ParticleSystem.current_displacement=}; displace_particle_system called multiple times without un-displacing. un-displacing is now broken.")
        self.ParticleSystem.current_displacement = displacement

        qx, qy, qz, *_ = displacement
        locations, _ = PS.x_v_current_3D

        # To apply rotations around COM we need to place it at the origin first
        COM = self.find_center_of_mass()
        self.translate_mesh(locations, -COM)

        new_locations = self.rotate_mesh(locations, displacement[3:])
        new_locations = self.translate_mesh(new_locations, displacement[:3])

        # Put back system in original location
        new_locations = self.translate_mesh(new_locations, COM)

        for i, location in enumerate(new_locations):
            # 'Unsafe' update needed to move fixed particles as well
            self.ParticleSystem.particles[i].update_pos_unsafe(location)


    def un_displace_particle_system(self):
        """
        Reverses current mesh displacement of the associated particle system.

        """

        if not hasattr(self.ParticleSystem, 'current_displacement'):
            raise AttributeError("Particle System is not currently displaced")

        elif type(self.ParticleSystem.current_displacement) == type(None):
            raise AttributeError("Particle System is not currently displaced")

        PS = self.ParticleSystem
        current_displacement = self.ParticleSystem.current_displacement
        reverse_displacement = -np.array(current_displacement)

        qx, qy, qz, *_ = reverse_displacement
        locations, _ = PS.x_v_current_3D

        # To apply rotations around COM we need to place it at the origin first
        COM = self.find_center_of_mass()
        self.translate_mesh(locations, -COM)

        # Extra syntax is to apply rotations in reverse order
        new_locations = self.rotate_mesh(locations, reverse_displacement[3:][::-1], order = 'xyz')
        new_locations = self.translate_mesh(new_locations, reverse_displacement[:3])

        # Put back system in original location
        new_locations = self.translate_mesh(new_locations, COM)

        for i, location in enumerate(new_locations):
            # 'Unsafe' update needed to move fixed particles as well
            self.ParticleSystem.particles[i].update_pos_unsafe(location)

        self.ParticleSystem.current_displacement = None

    def find_center_of_mass(self):
        """
        finds coordinates of center of mass of current mesh

        Returns
        -------
        COM : npt.ArrayLike
            [x,y,z] vector of center of mass

        """
        PS = self.ParticleSystem
        locations, _ = PS.x_v_current_3D
        masses = np.array([p.m for p in PS.particles])
        total_mass = np.sum(masses)
        weighing_vector = masses/total_mass
        for i in range(3):
            locations[:,i]*=weighing_vector
        COM = np.sum(locations,axis=0)
        return COM

    def translate_mesh(self, mesh, translation):
        """
        Translates mesh locations

        Parameters
        ----------
        mesh : npt.ArrayLike
            shape n x 3 array holding x, y, z locations of each point
        translation : list
            x, y, z axis translations

        Returns
        -------
        mesh : npt.ArrayLike
            shape n x 3 array holding x, y, z locations of each point

        """
        qx, qy, qz = translation

        mesh[:,0] += qx
        mesh[:,1] += qy
        mesh[:,2] += qz

        return mesh

    def rotate_mesh(self, mesh : npt.ArrayLike, rotations : list, order = 'zyx'):
        """
        Rotates mesh locations

        Parameters
        ----------
        mesh : npt.ArrayLike
            shape n x 3 array holding x, y, z locations of each point
        rotations : list
            x, y, z axis rotation angles in degrees

        Returns
        -------
        rotated_mesh : npt.ArrayLike
            shape n x 3 array holding x, y, z locations of each point

        """
        gamma, beta, alpha = rotations
        rotation_matrix = Rotation.from_euler(order, [alpha, beta, gamma], degrees=True)
        rotated_mesh = np.matmul(rotation_matrix.as_matrix(), mesh.T).T
        return rotated_mesh

    def calculate_restoring_forces(self):
        """
        calculates net forces and moments around the center of mass

        Returns
        -------
        net_force : npt.ArrayLike
            Net force on center of mass.
        net_moments : npt.ArrayLike
            Net moments around center of mass.

        """
        PS = self.ParticleSystem
        forces = self.force_value()
        net_force = np.sum(forces,axis=0)

        COM = self.find_center_of_mass()
        locations, _ = PS.x_v_current_3D
        moment_arms = self.translate_mesh(locations, -COM) # note: this doesn't displace the PS, just applies a transformation on the 'locations' variable
        moments = np.cross(moment_arms, forces)
        net_moments = np.sum(moments,axis=0)

        return net_force, net_moments

class ParticleOpticalPropertyType(Enum):
    """
    Enumeration representing the various types of optical properties for the Particles

    Attributes
    ----------
    SPECULAR : str
        Indicates that the particle reflects light specularly
    ARBITRARY_PHC : str
        Indicates that the particle represents an arbitrary photonic crystal
        NOTE: scipy.interpolate.interpnd.LinearNDInterpolator has to be set on
        particle.optical_interpolator(elevation, azimuth, polarisation_angle)
        ->(elevation, azimuth, magnitude)
    AXICONGRATING : str
        Indicates that the particle scatter light like a cone
        NOTE: Directing angle should be set in the format of a rotation matrix
        for the relevant particles that represents [rx, ry] rotations of area
        vector on property particle.axicon_angle
    """

    SPECULAR = "specular"
    AXICONGRATING = "axicongrating"
    ARBITRARY_PHC = "ARBITRARY_PHC"

vectorized_optical_type_retriever = np.vectorize(lambda  p: p.optical_type)

def compute_spherical_coordinates(area_vectors: npt.NDArray,
                                  polarisation_vectors: npt.NDArray) -> (
                                      npt.NDArray, npt.NDArray, npt.NDArray):
    """
    Computes the polar angles, azimuth angles, and polarisation angles of the incoming ray
    and its polarisation relative to the orientation of area elements represented by area vectors.

    Parameters
    ----------
    area_vectors : npt.NDArray
        An array of shape (n_particles, 3) representing the area vectors of n particles.
    polarisation_vectors : npt.NDArray
        An array of shape (n_particles, 2) representing the polarisation vectors of laser beams for
        n particles.

    Returns
    -------
    polar_angles : npt.NDArray
        An array of polar angles of the area vectors relative to the z-axis [rad].
    azimuth_angles : npt.NDArray
        An array of azimuth angles of the area vectors in the xy-plane [rad].
    polarisation_angles : npt.NDArray
        An array of angles between the polarisation vectors and their projection onto the plane
        orthogonal to the area vectors [rad].
    """
    # Normalize the area vectors
    norm_area_vectors = area_vectors / np.linalg.norm(area_vectors, axis=1)[:, np.newaxis]

    # Compute polar angles using the dot product between area vectors and the z-axis
    polar_angles = np.arccos(norm_area_vectors[:, 2])

    # Compute azimuth angles
    azimuth_angles = np.arctan2(norm_area_vectors[:,1], norm_area_vectors[:,0])

    # Compute the polarisation angle in cartesian space:
    polarisation_angles = np.arccos(polarisation_vectors[:,0])


    return polar_angles, azimuth_angles, polarisation_angles


def spherical_to_cartesian(polar_angles: npt.NDArray,
                           azimuth_angles: npt.NDArray,
                           magnitudes: npt.NDArray) -> npt.NDArray:
    """
    Converts spherical coordinates back to Cartesian coordinates in the global frame,
    using the area vectors to define the local reference frames. Scales the resulting vectors by the magnitudes.

    Parameters
    ----------
    polar_angles : npt.NDArray
        An array of polar angles in radians.
    azimuth_angles : npt.NDArray
        An array of azimuth angles in radians.
    magnitudes : npt.NDArray
        An array of magnitudes to scale the intensity of the resulting rays.
    area_vectors : npt.NDArray
        An array of shape (n_particles, 3) representing the area vectors of n particles,
        used to define the local reference frames.

    Returns
    -------
    cartesian_vectors : npt.NDArray
        An array of shape (n_particles, 3) representing the resulting Cartesian vectors in the global frame.
    """
    # Convert spherical to Cartesian coordinates in the local frame
    x = magnitudes * np.sin(polar_angles) * np.cos(azimuth_angles)
    y = magnitudes * np.sin(polar_angles) * np.sin(azimuth_angles)
    z = magnitudes * np.cos(polar_angles)

    cartesian_vectors= np.vstack((x,y,z)).T

    return cartesian_vectors

def cartesian_to_sphereical(vectors:npt.NDArray) -> npt.NDArray:
    """
    Converts a set of vectors from cartesian to spherical coordinates

    Parameters
    ----------
    vectors : npt.NDArray
        An array of shape (n_particles, 3) representing the vectors to convert.

    Returns
    -------
    polar_angles : npt.NDArray
        An array of polar angles in radians.
    azimuth_angles : npt.NDArray
        An array of azimuth angles in radians.
    magnitudes : npt.NDArray
        An array of magnitudes of the vectors.
    """
    x,y,z = vectors[:,0], vectors[:,1], vectors[:,2]
    magnitudes = np.linalg.norm(vectors, axis=1)
    polar_angles = np.arccos(z/magnitudes)
    azimuth_angles = np.arctan2(y,x)
    return polar_angles, azimuth_angles, magnitudes

def wrap_spherical_coordinates(theta: npt.NDArray,
                               phi: npt.NDArray,
                               pol: npt.NDArray=None):

    """
    wraps points in spherical coordinates to always stay within the interpolators defined range

    Parameters
    ----------
    theta : npt.NDArray
        polar angle [rad]
    phi : npt.NDArray
        azimuthal angle [rad]
    pol : npt.NDArray
        polarisation angle [rad]


    Returns
    -------
    theta : npt.NDArray
        polar angle
    phi : npt.NDArray
        azimuthal angle
    pol : npt.NDArray
        polarisation angle
    """
    phi[theta>np.pi] += np.pi
    theta[theta>np.pi]= np.pi - theta[theta>np.pi]%np.pi

    phi[theta<0] += np.pi
    theta[theta<0] *=-1
    phi %= 2*np.pi
    
    phi[abs(phi-2*np.pi)<1e-5]=0 # wraps values that are _almost_ 2*np.pi

    if np.any(pol != None):
        x = abs(np.cos(pol))
        y = abs(np.sin(pol))
        pol = np.arctan(y/x)

        return theta, phi, pol
    else:
        return theta, phi

    # quick test
    # !!! todo move to testing file
    theta = np.random.random(100)*3*np.pi - np.pi
    phi = np.random.random(100)*3*np.pi - np.pi
    pol = np.random.random(100)*3*np.pi - np.pi
    def test_wrap_spherical_coordinates(dat):
        mags = np.ones(dat[0].shape)
        t1 = spherical_to_cartesian(*wrap_spherical_coordinates(*dat)[:2], mags)
        t2 = spherical_to_cartesian(*dat[:2], mags)
        return (t1==t2).all()
    test_wrap_spherical_coordinates((theta,phi,pol))

if __name__ == "__main__":
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

    # One half of example will be 45 deg axicon angle directed towards (5,5)
    # other half will be specular reflection
    rots = []

    for particle in PS.particles:
        particle.optical_type = ParticleOpticalPropertyType.SPECULAR

        if (particle.x[0]-5)**2 + (particle.x[1]-5)**2>= 3**2:
            roty = 45
            rotz = np.rad2deg(np.arctan2((particle.x[1]-5), (particle.x[0]-4.999)))
            particle.optical_type = ParticleOpticalPropertyType.AXICONGRATING
            #particle.axicon_angle = Rotation.from_euler('yz', [roty, rotz], degrees=True).as_matrix()
            particle.axicon_angle = Rotation.from_euler('yz', [roty, rotz], degrees=True).as_matrix()
            rots.append((roty,rotz%360))

    OFC = OpticalForceCalculator(PS, LB)

    forces = OFC.force_value()


    ax = PS.plot()

    points, _ = PS.x_v_current_3D
    x,y,z = points[:,0], points[:,1], points[:,2]
    a_u = forces[:,0]
    a_v = forces[:,1]
    a_w = forces[:,2]
    ax.scatter(x,y,z)
    ax.quiver(x,y,z,a_u,a_v,a_w, length = 0.1)
    ax.set_box_aspect([1,1,1])
    ax.set_zlim(-5, 5)
    #ax.set_zscale('symlog')

    #ax2 = fig.add_subplot(projection='3d')
    #LB.plot(ax2, x_range = [0,10], y_range=[0,10])









