"""
Input file for validation of PS, benchmark case where form of self-stressed saddle is sought
"""
import numpy as np

# grid discretization
grid_size = 5
grid_length = 10
grid_height = 5


def connectivity_matrix(grid_size: int):
    n = grid_size ** 2 + (grid_size - 1) ** 2
    top_edge = [i for i in range(grid_size)]
    bottom_edge = [n - grid_size + i for i in range(grid_size)]
    left_edge = [(grid_size * 2 - 1) * i for i in range(1, grid_size - 1)]
    right_edge = [left_edge[i] + grid_size - 1 for i in range(grid_size - 2)]
    fixed_nodes = top_edge + bottom_edge + left_edge + right_edge

    connections = []

    # inner grid connections
    for i in range(n):
        if i not in fixed_nodes:
            connections.append([i, i - grid_size])
            connections.append([i, i - grid_size + 1])
            connections.append([i, i + grid_size - 1])
            connections.append([i, i + grid_size])

    matrix = np.zeros((n, n))

    for indices in connections:                 # enter connections at correct index in matrix
        matrix[indices[0], indices[1]] += 1
        matrix[indices[1], indices[0]] += 1

    matrix[matrix > 1] = 1                      # remove double connections

    b = np.nonzero(np.triu(matrix))
    b = np.column_stack((b[0], b[1]))

    return b, fixed_nodes


def initial_conditions(g_size: int, m_segment: float, fixed_nodes: list, g_h: float, g_l: float):
    conditions = []

    orthogonal_distance = g_l/(g_size-1)
    dl = g_h / g_l * orthogonal_distance
    even = [i * orthogonal_distance for i in range(g_size)]
    uneven = [i*orthogonal_distance + 0.5*orthogonal_distance for i in range(g_size - 1)]
    x_y = [[i * orthogonal_distance, 0] for i in range(g_size)]

    for i in range(g_size - 1):
        x_y.extend(list(zip(uneven, [i*orthogonal_distance + 0.5*orthogonal_distance for j in range(len(uneven))])))
        x_y.extend(list(zip(even, [(i+1)*orthogonal_distance for j in range(len(even))])))

    z = [i*dl for i in range(g_size)]
    temp = z.copy()
    z.extend(reversed(temp))
    z.extend(temp[1:-1])
    z.extend(reversed(temp[1:-1]))

    n = g_size ** 2 + (g_size - 1) ** 2

    for i in range(n):
        if i in fixed_nodes:
            conditions.append([list(x_y[i]) + [z[fixed_nodes.index(i)]], [0, 0, 0], m_segment, True])
        else:
            conditions.append([list(x_y[i]) + [g_h/2], [0, 0, 0], m_segment, False])

    return conditions


def element_parameters(k, c, c_m, i_c):
    e_m = []

    for indeces in c_m:
        i0, i1 = indeces[0], indeces[1]
        # l0 = np.linalg.norm(np.array(i_c[i0][0]) - np.array(i_c[i1][0]))
        e_m.append([k, 0, c])

    return e_m


# dictionary of required parameters
params = {
    # model parameters
    "n": 10,  # [-]       number of particles
    "k_t": 1,  # [N/m]     spring stiffness
    "c": 1,  # [N s/m] damping coefficient
    "L": 10,  # [m]       tether length
    "m_block": 100,  # [kg]     mass attached to end of tether
    "rho_tether": 0.1,  # [kg/m]    mass density tether

    # simulation settings
    "dt": 0.1,  # [s]       simulation timestep
    "t_steps": 1000,  # [-]      number of simulated time steps
    "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
    "max_iter": 1e5,  # [-]       maximum number of iterations

    # physical parameters
    "g": 9.807,  # [m/s^2]   gravitational acceleration
    "v_w": [5, 0, 0],  # [m/s]     wind velocity vector
    'rho': 1.225,  # [kg/ m3]  air density
    'c_d_bridle': 1.05,  # [-]       drag-coefficient of bridles
    "d_bridle": 0.02  # [m]       diameter of bridle lines
}

# calculated parameters
params["l0"] = 0#np.sqrt( 2 * (grid_length/(grid_size-1))**2)
params["m_segment"] = 1
params["k"] = params["k_t"] * (params["n"] - 1)  # segment stiffness
params["n"] = grid_size ** 2 + (grid_size - 1) ** 2


# instantiate connectivity matrix and initial conditions array
c_matrix, f_nodes = connectivity_matrix(grid_size)
init_cond = initial_conditions(grid_size, params["m_segment"], f_nodes, grid_height, grid_length)
elem_params = element_parameters(params["k"], params["c"], c_matrix, init_cond)

# print(init_cond)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # from mpl_toolkits import mplot3d

    x = []
    y = []
    z = []
    for i in range(len(init_cond)):
        x.append(init_cond[i][0][0])
        y.append(init_cond[i][0][1])
        z.append(init_cond[i][0][2])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # b = np.nonzero(np.triu(c_matrix))
    # b = np.column_stack((b[0], b[1]))

    ax.scatter(x, y, z, c='red')
    for indices in c_matrix:
        ax.plot([x[indices[0]], x[indices[1]]], [y[indices[0]], y[indices[1]]], [z[indices[0]], z[indices[1]]],
                color='black')

    # ax.plot(x, z, 'r+', zdir='y', zs=-1.5)
    # ax.plot(y, z, 'g+', zdir='x', zs=-0.5)
    # ax.plot(x, y, 'k+', zdir='z', zs=-1.5)

    plt.show()

    print(len(initial_conditions(5, 1, [0, 1, 2, 3, 4, 36, 37, 38, 39, 40, 9, 18, 27, 13, 22, 31], 5, 10)))
