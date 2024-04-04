# -*- coding: utf-8 -*-
"""
Created on Fri Dec  15 12:27:53 2023

@author: Mark Kalsbeek
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.particleSystem.SpringDamper import SpringDamperType



params = {
    # model parameters
    "k": 1,  # [N/m]     spring stiffness
    "k_d": 1,  # [N/m]     spring stiffness
    "c": 1,  # [N s/m] damping coefficient
    "m_segment": 1, # [kg] mass of each node
    }


def mesh_square(length, width, mesh_edge_length, params = params):
    n_wide = int(width/ mesh_edge_length + 1)
    n_long = int(length/ mesh_edge_length + 1)

    mesh = np.meshgrid(np.linspace(0, length, n_long),
                       np.linspace(0, width, n_wide))

    initial_conditions = []
    xy_coordinates = np.column_stack(list(zip(mesh[0],mesh[1]))).T
    xyz_coordinates = np.column_stack((xy_coordinates,np.zeros(len(xy_coordinates)).T))

    for xyz in xyz_coordinates:
        initial_conditions.append([xyz, np.zeros(3), params['m_segment'], False])

    connections = []
    #We know that all the nodes are connected to those of the next row, which is grid_length+1 units further
    for i, node in enumerate(initial_conditions[:-n_long]): # adding connextions in y-axis
        connections.append([i, i+n_long, params['k'], params['c']])


    # We can do the same for the connections between the columns
    for i, node in enumerate(initial_conditions): # adding connections in x-axis
        if (i+1)%(n_long): # Using modulus operator to exclude the nodes at the end of a row
            connections.append([i, i+1, params['k'], params['c']])

    return initial_conditions, connections


def mesh_square_cross(length, width, mesh_edge_length, params = params):
    n_wide = int(width/ mesh_edge_length + 1)
    n_long = int(length/ mesh_edge_length + 1)

    mesh = np.meshgrid(np.linspace(0, length, n_long),
                       np.linspace(0, width, n_wide))

    initial_conditions = []
    xy_coordinates = np.column_stack(list(zip(mesh[0],mesh[1]))).T
    xyz_coordinates = np.column_stack((xy_coordinates,np.zeros(len(xy_coordinates)).T))

    for xyz in xyz_coordinates:
        initial_conditions.append([xyz, np.zeros(3), params['m_segment'], False])

    connections = []
    #We know that all the nodes are connected to those of the next row, which is grid_length+1 units further
    for i, node in enumerate(initial_conditions[:-n_long]): # adding connextions in y-axis
        connections.append([i, i+n_long, params['k'], params['c']])

        if (i+1)%(n_long): #cross connections
            connections.append([i, i+n_long+1, params['k_d'], params['c']])
            connections.append([i+1, i+n_long, params['k_d'], params['c']])

    # We can do the same for the connections between the columns
    for i, node in enumerate(initial_conditions): # adding connections in x-axis
        if (i+1)%(n_long): # Using modulus operator to exclude the nodes at the end of a row
            connections.append([i, i+1, params['k'], params['c']])

    return initial_conditions, connections

def mesh_square_cross_sparse(length, width, mesh_edge_length, params = params):
    n_wide = int(width/ mesh_edge_length + 1)
    n_long = int(length/ mesh_edge_length + 1)


    mesh = np.meshgrid(np.linspace(0, length, n_long),
                       np.linspace(0, width, n_wide))

    initial_conditions = []
    xy_coordinates = np.column_stack(list(zip(mesh[0],mesh[1]))).T
    xyz_coordinates = np.column_stack((xy_coordinates,np.zeros(len(xy_coordinates)).T))

    for xyz in xyz_coordinates:
        initial_conditions.append([xyz, np.zeros(3), params['m_segment'], False])

    connections = []
    #We know that all the nodes are connected to those of the next row, which is grid_length+1 units further
    flip = True
    for i, node in enumerate(initial_conditions[:-n_long]): # adding connextions in y-axis
        connections.append([i, i+n_long, params['k'], params['c']])

        if (i+1)%(n_long): #cross connections
            if flip:
                connections.append([i, i+n_long+1, params['k_d'], params['c']])
                flip = False
            else:
                connections.append([i+1, i+n_long, params['k_d'], params['c']])
                flip= True
        else:
            flip = not flip

    # We can do the same for the connections between the columns
    for i, node in enumerate(initial_conditions): # adding connections in x-axis
        if (i+1)%(n_long): # Using modulus operator to exclude the nodes at the end of a row
            connections.append([i, i+1, params['k'], params['c']])

    return initial_conditions, connections

def mesh_square_concentric(length, mesh_edge_length, params = params ,fix_outer = False):
    n_long = int(length/ mesh_edge_length + 1)

    x_space = np.linspace(-length/2, length/2, n_long)
    y_space = x_space

    mesh = np.meshgrid(x_space,y_space)

    initial_conditions = []
    xy_coordinates = np.column_stack(list(zip(mesh[0],mesh[1]))).T
    xyz_coordinates = np.column_stack((xy_coordinates,np.zeros(len(xy_coordinates)).T))

    for xyz in xyz_coordinates:
        if (abs(xyz[0]) == length/2 and abs(xyz[1]) == length/2) and fix_outer:
            initial_conditions.append([xyz, np.zeros(3), params['m_segment'], True])
        else:
            initial_conditions.append([xyz, np.zeros(3), params['m_segment'], False])

    connections = []
    dia_counter = [0,n_long-2]
    #We know that all the nodes are connected to those of the next row, which is grid_length+1 units further
    for i, node in enumerate(initial_conditions[:-n_long]): # adding connextions in y-axis
        if abs(node[0][0])>abs(node[0][1]):
            connections.append([i, i+n_long, params['k'], params['c']])

        if abs(node[0][0])>=abs(node[0][1]) and node[0][1]<0:
            connections.append([i, i+n_long, params['k'], params['c']])

        if dia_counter[0] == i: #cross connections
            dia_counter[0] += n_long +1
            connections.append([i, i+n_long+1, params['k_d'], params['c']])

        if dia_counter[1] == i:
            dia_counter[1] += n_long-1
            connections.append([i+1, i+n_long, params['k_d'], params['c']])

    # We can do the same for the connections between the columns
    for i, node in enumerate(initial_conditions): # adding connections in x-axis
        if (i+1)%(n_long) and abs(node[0][0])<=abs(node[0][1]) and node[0][0]<0: # Using modulus operator to exclude the nodes at the end of a row
            connections.append([i, i+1, params['k'], params['c']])
        elif (i+1)%(n_long) and abs(node[0][0])<abs(node[0][1]) and node[0][0]>=0: # Using modulus operator to exclude the nodes at the end of a row
            connections.append([i, i+1, params['k'], params['c']])


    return initial_conditions, connections

def mesh_airbag_square_cross(length, width= 0, mesh_edge_length = 1/10,  params = params, noncompressive = False, sparse = False):

    if sparse:
        meshfunct = mesh_square_cross_sparse
    else:
        meshfunct = mesh_square_cross

    if width==0:
        width = length

    initial_conditions, connections = meshfunct(length,
                                                width,
                                                mesh_edge_length,
                                                params)

    # We iterate over the particle and set specific constraint conditions
    # to match the symmetry of the airbag being cut into 8 pieces
    for particle in initial_conditions:
        # sequence of if/elif statements matters becuase some of the used
        # critera can override others.


        # Fixing s.t. center node only move in z axis
        if (particle[0] == [0,0,0]).all():
            particle[3] = True
            particle.append([0,0,1])
            particle.append('line')

        elif particle[0][0] == length and particle[0][1] == 0:
            particle[3] = True
            particle.append([1,0,0])
            particle.append('line')

        elif particle[0][0] == 0 and particle[0][1] == length:
            particle[3] = True
            particle.append([0,1,0])
            particle.append('line')

        elif    (
                (particle[0][0] == length and particle[0][1]>0)
                or
                (particle[0][1] == width and particle[0][0]>0)
                ):
            particle[3] = True
            particle.append([0,0,1])
            particle.append('plane')

        elif particle[0][0] == 0 and particle[0][1]>0 and particle[0][1]<length:
            particle[3] = True
            particle.append([1,0,0])
            particle.append('plane')

        elif particle[0][1] == 0 and particle[0][0]>0 and particle[0][0]<length:
            particle[3] = True
            particle.append([0,1,0])
            particle.append('plane')

    if noncompressive:
        linktype = SpringDamperType.NONCOMPRESSIVE

        for link in connections:
            link.append(linktype)

    return initial_conditions, connections

def mesh_phc_square_cross(length, width= 0, mesh_edge_length = 1/10, params = params, noncompressive = False, sparse = False):
    required = ['E_x', 'E_y', "G", "thickness"]

    for key in required:
        if not key in params.keys():
            raise KeyError(f"{key} missing from params")

    if width==0:
        width = length

    n_wide = int(width/ mesh_edge_length + 1) # x count
    n_long = int(length/ mesh_edge_length + 1) # y count
    x_length = width/n_wide
    y_length = length/n_long

    # Calculate k's from the given stifnesses
    # First calculate the diagnonal, it is required in the calculation of the orthogonal ones
    params["k_d"] = params["G"] * params["thickness"] * x_length / (y_length*n_wide/np.sqrt(2))
    params["k_x"] = params["E_x"] * params["thickness"]
    params["k_y"] = params["E_y"] * params["thickness"]
    # Now we have to reduce the influence of the orthogonal springs in order to account for the
    # contribution of the diagonal ones
    params["k_x"]*= params["k_x"]*n_long / (params["k_x"]*n_long + params["k_d"]*np.sqrt(2)*n_long)
    params["k_y"]*= params["k_y"]*n_wide / (params["k_y"]*n_wide + params["k_d"]*np.sqrt(2)*n_wide)

    # Perform the meshing
    mesh = np.meshgrid(np.linspace(0, length, n_long),
                       np.linspace(0, width, n_wide))

    initial_conditions = []
    xy_coordinates = np.column_stack(list(zip(mesh[0],mesh[1]))).T
    xyz_coordinates = np.column_stack((xy_coordinates,np.zeros(len(xy_coordinates)).T))

    for xyz in xyz_coordinates:
        initial_conditions.append([xyz, np.zeros(3), params['m_segment'], False])

    connections = []
    #We know that all the nodes are connected to those of the next row, which is grid_length+1 units further
    for i, node in enumerate(initial_conditions[:-n_long]): # adding connextions in y-axis
        connections.append([i, i+n_long, params['k_y'], params['c']])

        if (i+1)%(n_long): #cross connections
            connections.append([i, i+n_long+1, params['k_d'], params['c']])
            connections.append([i+1, i+n_long, params['k_d'], params['c']])

    # We can do the same for the connections between the columns
    for i, node in enumerate(initial_conditions): # adding connections in x-axis
        if (i+1)%(n_long): # Using modulus operator to exclude the nodes at the end of a row
            connections.append([i, i+1, params['k_x'], params['c']])

    if noncompressive:
        linktype = SpringDamperType.NONCOMPRESSIVE

        for link in connections:
            link.append(linktype)

    return connections, initial_conditions


def mesh_circle_square_cross(radius, mesh_edge_length, params = params, fix_outer = False, edge = 0):
    n_wide = int(radius/ mesh_edge_length + 1)
    n_long = n_wide

    mesh = np.meshgrid(np.linspace(-radius, radius, n_long),
                       np.linspace(-radius, radius, n_wide))

    initial_conditions = []
    xy_coordinates = np.column_stack(list(zip(mesh[0],mesh[1]))).T
    xyz_coordinates = np.column_stack((xy_coordinates,np.zeros(len(xy_coordinates)).T))

    for xyz in xyz_coordinates:
        initial_conditions.append([xyz, np.zeros(3), params['m_segment'], False])

    connections = []
    #We know that all the nodes are connected to those of the next row, which is grid_length+1 units further
    for i, node in enumerate(initial_conditions[:-n_long]): # adding connextions in y-axis
        connections.append([i, i+n_long, params['k'], params['c']])

        if (i+1)%(n_long): #cross connections
            connections.append([i, i+n_long+1, params['k_d'], params['c']])
            connections.append([i+1, i+n_long, params['k_d'], params['c']])

    # We can do the same for the connections between the columns
    for i, node in enumerate(initial_conditions): # adding connections in x-axis
        if (i+1)%(n_long): # Using modulus operator to exclude the nodes at the end of a row
            connections.append([i, i+1, params['k'], params['c']])

    # Now to trim the excess nodes and connections
    mask = xy_coordinates[:,0]**2 + xy_coordinates[:,1]**2 <= radius**2


    dumplist = []

    for i, keepit in enumerate(mask):
        if not keepit:
            initial_conditions[i][3]= True
            # print(f'Iterating point {i}')
            for j, link in enumerate(connections):
                if i in link[:2]:
                    # print(f'dumping link {j} for being connected to {i}: {link}')
                    dumplist.append(j)

    dumplist.sort()
    dumplist = list(set(dumplist))[::-1]
    # print(f'dumplist length: {len(set(dumplist))}\n{dumplist[::-1]}')
    # print(f'connections length: {len(connections)}')
    for i in dumplist:
        del connections[i]


    #for i, item in enumerate(initial_conditions):
    #    if mask[i]:
    #        del initial_conditions[i]

    if fix_outer:
        if edge == 0:
            edge = mesh_edge_length * 1.5
        inner = xy_coordinates[:,0]**2 + xy_coordinates[:,1]**2 <= (radius-edge)**2
        for i, freeit in enumerate(inner):
            if not freeit:
                initial_conditions[i][3]= True

    return initial_conditions, connections

def mesh_rotate_and_trim(initial_conditions, connections, angle):
    """
    NOTE: Input mesh is expected to be square!
    """
    center_of_mass = np.array([0,0,0],dtype ='float64')
    for particle in initial_conditions:
        center_of_mass+=particle[0]
    center_of_mass = center_of_mass / len(initial_conditions)

    x_cleaned = np.array([i[0] for i in initial_conditions])
    x_range, y_range, z_range = np.ptp(x_cleaned, axis = 0)

    # rotation shrinks size of inscribed rectangle
    # Going for constant angle for consistent size
    factor = np.cos(np.deg2rad(45))
    x_range *= factor
    y_range *= factor

    rotation_matrix = R.from_euler('z', angle, degrees = True).as_matrix()

    for particle in initial_conditions:
        particle[0] -= center_of_mass

        particle[0] = rotation_matrix.dot(particle[0])

    dumplist = set()
    for i, link in enumerate(connections):
        xyz_0 = initial_conditions[link[0]][0]
        xyz_1 = initial_conditions[link[1]][0]
        if abs(xyz_0[0])> x_range/2:
            dumplist.add(i)
        elif abs(xyz_0[1])> y_range/2:
            dumplist.add(i)
        elif abs(xyz_1[0])> x_range/2:
            dumplist.add(i)
        elif abs(xyz_1[1])> y_range/2:
            dumplist.add(i)
    dumplist = list(dumplist)
    dumplist.sort()

    for i in dumplist[::-1]:
        del connections[i]

    return initial_conditions, connections


def ps_fix_opposite_boundaries_x(ParticleSystem, margin = 0.075):
    """
    Fixes two boundaries in preparation for unidirectional pull test

    """
    center_of_mass = np.array([0,0,0],dtype ='float64')
    for particle in ParticleSystem.particles:
        center_of_mass+=particle.x
    center_of_mass = center_of_mass / len(ParticleSystem.particles)

    x_cleaned = np.array([particle.x[0] for particle in ParticleSystem.particles])
    x_range = np.ptp(x_cleaned, axis = 0)

    for particle in ParticleSystem.particles:
        particle.update_pos_unsafe(particle.x-center_of_mass)

    boundary_x_min = []
    boundary_x_plus = []

    for i, particle in enumerate(ParticleSystem.particles):
        if abs(particle.x[0]) > ((x_range/2)*(1-margin)):
            particle.set_fixed(True)

            if particle.x[0]>0:
                boundary_x_plus.append(i)
            else:
                boundary_x_min.append(i)
    boundaries = [boundary_x_min, boundary_x_plus]

    return ParticleSystem, boundaries

def ps_stretch_in_x(ParticleSystem, boundary, displacement):
    for indice in boundary:
        particle = ParticleSystem.particles[indice]
        new_pos = particle.x
        new_pos[0] += displacement
        particle.update_pos(new_pos)

def ps_find_reaction_of_boundary(ParticleSystem, boundary):
    # !!! ATTENTION !!! DRAFT CODE! COMPLETLY UNTESTED!
    internal_forces = ParticleSystem._ParticleSystem__one_d_force_vector()
    reaction = np.array([0.0, 0.0, 0.0])
    for indice in boundary:
        reaction += internal_forces[indice*3: indice*3+3]
    return reaction

def ps_find_mid_strip_y(ParticleSystem, width= 1):
    center_of_mass = np.mean(ParticleSystem.x_v_current_3D[0], axis=0)
    for particle in ParticleSystem.particles:
        particle.update_pos_unsafe(particle.x-center_of_mass)

    midstrip = []
    for i, particle in enumerate(ParticleSystem.particles):
        pos = particle.x
        if abs(pos[0]) <= width/2:
            midstrip.append(i)
    return midstrip

def ps_find_strip_dimentions(ParticleSystem, midstrip):
    positions = []
    for indice in midstrip:
        particle = ParticleSystem.particles[indice]
        positions.append(particle.x)
    positions = np.array(positions)
    point_to_point_range = np.ptp(positions, axis = 0)
    return point_to_point_range

if __name__ == '__main__':
    from src.particleSystem.ParticleSystem import ParticleSystem
    import matplotlib.pyplot as plt
    params = {
        # model parameters
        "k": 1,  # [N/m]   spring stiffness
        "k_d": 1,  # [N/m] spring stiffness for diagonal elements
        "c": 1,  # [N s/m] damping coefficient
        "m_segment": 1, # [kg] mass of each node
        # simulation settings
        "dt": 0.1,  # [s]       simulation timestep
        "t_steps": 1000,  # [-]      number of simulated time steps
        "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
        "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
        "max_iter": 1e5,  # [-]       maximum number of iterations]
        }

    # !!! Don't forget to add new meshing functions to this list!
    meshing_functions = [mesh_square,
                         mesh_square_cross,
                         mesh_square_cross_sparse,
                         mesh_airbag_square_cross,
                         mesh_square_concentric,
                         mesh_circle_square_cross]
    inputs = [16,8,1, params]
    nplots = len(meshing_functions) + 1

    cols = int(np.sqrt(nplots)) +1
    rows = nplots // cols
    if nplots % cols !=0:
        rows +=1


    fig = plt.figure()
    pslist = []
    for i, function in enumerate(meshing_functions):
        ax = fig.add_subplot(rows, cols,i+1,projection='3d')
        ax.set_box_aspect([1,1,1])
        if i ==4:
            inputs = inputs[1:]
        initial_conditions, connections = function(*inputs)
        PS = ParticleSystem(connections, initial_conditions, params)
        pslist.append(PS)

        PS.plot(ax)
        ax.set_title(function.__name__)


    initial_conditions, connections = mesh_square_cross(20,20,1,params)
    initial_conditions, connections = mesh_rotate_and_trim(initial_conditions,
                                                           connections,
                                                           45/2)
    PS = ParticleSystem(connections, initial_conditions,params)
    PS, boundaries = ps_fix_opposite_boundaries_x(PS, margin = 0.175)

    ps_stretch_in_x(PS, boundaries[1], 1)

    pslist.append(PS)

    ax = fig.add_subplot(rows, cols, nplots, projection='3d')
    PS.plot(ax)
    ax.set_title((mesh_square_cross.__name__, mesh_rotate_and_trim.__name__, ps_fix_opposite_boundaries_x.__name__, ps_stretch_in_x.__name__))


