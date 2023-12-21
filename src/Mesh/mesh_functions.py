# -*- coding: utf-8 -*-
"""
Created on Fri Dec  15 12:27:53 2023

@author: Mark Kalsbeek
"""
import numpy as np
import scipy as sp


params = {
    # model parameters
    "k": 1,  # [N/m]     spring stiffness
    "c": 1,  # [N s/m] damping coefficient
    "m_segment": 1, # [kg] mass of each node
    }


def mesh_square(length, width, mesh_edge_length):
    aspect_ratio = length/width
    n_wide = int(width/ mesh_edge_length + 1)
    n_long = int(length/ mesh_edge_length + 1)
    n_points = n_wide*n_long

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


def mesh_square_cross(length, width, mesh_edge_length):
    aspect_ratio = length/width
    n_wide = int(width/ mesh_edge_length + 1)
    n_long = int(length/ mesh_edge_length + 1)
    n_points = n_wide*n_long

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


def mesh_square_concentric(length, width, mesh_edge_length, fix_outer = False):
    n_wide = int(width/ mesh_edge_length + 1)
    n_long = int(length/ mesh_edge_length + 1)
    
    x_space = np.linspace(-length/2, length/2, n_long)
    y_space = np.linspace(-width/2, width/2, n_wide)
    
    mesh = np.meshgrid(x_space,y_space)
    
    initial_conditions = []
    xy_coordinates = np.column_stack(list(zip(mesh[0],mesh[1]))).T
    xyz_coordinates = np.column_stack((xy_coordinates,np.zeros(len(xy_coordinates)).T))
    
    for xyz in xyz_coordinates:
        if (abs(xyz[0]) == length/2 and abs(xyz[1]) == width/2) and fix_outer:
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

def mesh_circle_square_cross(radius, mesh_edge_length, fix_outer = False, edge = 0):
    n_wide = int(radius/ mesh_edge_length + 1)
    n_long = n_wide
    n_points = n_wide*n_long

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
    
    meshing_functions = [mesh_square, mesh_square_cross, mesh_square_concentric, mesh_circle_square_cross]
    inputs = [16,16,1]
    nplots = len(meshing_functions)
    
    fig = plt.figure()
    pslist = []
    for i, function in enumerate(meshing_functions):
        ax = fig.add_subplot(1, nplots,i+1,projection='3d')
        if i ==3:
            inputs = inputs[1:]
        initial_conditions, connections = function(*inputs)
        PS = ParticleSystem(connections, initial_conditions,params)
        pslist.append(PS)
        PS.plot(ax)