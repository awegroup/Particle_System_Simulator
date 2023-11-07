"""
Input file for validation of PS, Hencky problem
"""
import numpy as np

filename = 'hencky_mesh_finer_mesh.msh'  # mesh file

# Matlab code to calculate value of b_0, as Python's sympy library is way too slow to calculate numeric values:
"""
    clear all
    clc

    syms b_0
    mu = 0;           % [-] Poisson's ratio
    eqn = 0 == (1-mu) * b_0 - (3-mu) / (b_0 ^ 2) - (5 - mu) * 2 / (3 * b_0 ^ 5) - (7 - mu) * 13 / (18 * b_0 ^ 8)     - (9 - mu) * 17 / (18 * b_0 ^ 11) - (11 - mu) * 37 / (27 * b_0 ^ 14) - (13 - mu) * 1205 / (567 * b_0 ^ 17)     - (15 - mu) * 219241 / (63504 * b_0 ^ 20) - (17 - mu) * 6634069 / (1143072 * b_0 ^ 23) - (19 - mu) * 51523763 / (5143824 * b_0 ^ 26) - (21 - mu) * 998796305 / (56582064 * b_0 ^ 29);
    sol = solve(eqn, b_0);

    numeric_sol = double(sol);
    tolerance = 1e-10;  % Define a small tolerance
    real_solutions = numeric_sol(abs(imag(numeric_sol)) < tolerance)
"""

# mu = Poisson's ratio
# precalculated values of b_0:
# mu,   b_0
# 0.0,  1.6204
# 0.1,  1.6487
# 0.2,  1.6827
# 0.3,  1.7244
# 0.34, 1.7439
# 0.4,  1.7769

b_0 = 1.6204
r = 0.1425  # [m] radius circular membrane
p = 100  # [kPa] uniform transverse pressure
E_t = 311488  # [N/m] Young's modules membrane material
d = 10  # [m] Membrane thickness

q = (p * r) / E_t

# for n = 10, first 11 relations of the a_2n parameter
a0 = 1 / b_0
a2 = 1 / (2 * b_0 ** 4)
a4 = 5 / (9 * b_0 ** 7)
a6 = 55 / (72 * b_0 ** 10)
a8 = 7 / (6 * b_0 ** 13)
a10 = 205 / (108 * b_0 ** 16)
a12 = 17051 / (5292 * b_0 ** 19)
a14 = 2864485 / (508032 * b_0 ** 22)
a16 = 103863265 / (10287648 * b_0 ** 25)
a18 = 27047983 / (1469664 * b_0 ** 28)
a20 = 42367613873 / (1244805408 * b_0 ** 31)
a = [a0, a2, a4, a6, a8, a10, a12, a14, a16, a18, a20]


def cm_and_ic(mesh_file, m, E, c):
    coordinates = []
    connections = []
    fixed = 40  # write automation later when I have more experience with the formatting of .msh files
    n = 0

    # read mesh file
    with open(mesh_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.startswith("$Nodes"):  # retrieve nodal coordinates
            entity_bloc, nodes_total, min_node_tag, max_node_tag = lines[i + 1].split()
            n = int(nodes_total)
            total_lines = (int(entity_bloc) + int(nodes_total)) * 2 + 1
            for j in range(1, total_lines):
                if len(lines[i + j].split()) == 3:
                    coordinate = lines[i + j].split()
                    coordinates.append([float(coordinate[i]) for i in range(3)])

        if line.startswith("$Elements"):  # retrieve nodal connections
            entity_bloc, nodes_total, min_node_tag, max_node_tag = lines[i + 1].split()
            total_lines = int(entity_bloc) + int(nodes_total) + 2

            for j in range(1, total_lines):
                if len(lines[i + j].split()) != 4:
                    connection = lines[i + j].split()
                    connections.append([int(connection[i]) for i in range(1, len(connection))])

    i_c = []  # construct initial conditions matrix: [x, v, m, fixed]
    for i in range(n):
        if i < fixed:
            i_c.append([coordinates[i], [0, 0, 0], m, True])
        else:
            i_c.append([coordinates[i], [0, 0, 0], m, False])

    c_m = []  # construct connectivity matrix: [[index p1, index p2], ...]
    for element in connections:
        for i in range(len(element)):
            if i + 1 == len(element):
                c_m.append([element[i] - 1, element[0] - 1])
            else:
                c_m.append([element[i] - 1, element[i + 1] - 1])
    c_m = list(set(tuple(sorted(pair)) for pair in c_m))
    c_m = [list(pair) for pair in c_m if pair[0] != pair[1]]

    e_p = []  # construct element parameter array: [k, l0, c]
    for nodes in c_m:
        node1, node2 = nodes[0], nodes[1]
        A = 0
        for element in connections:
            if node1 + 1 in element and node2 + 1 in element and len(element) > 2:
                # print("nodes:", nodes)
                # print("element:", element)
                p1 = element.index(node1 + 1)
                p2 = element.index(node2 + 1)
                all_indices = list(range(len(element)))
                all_indices.remove(p1)
                all_indices.remove(p2)
                p3, p4 = all_indices
                # print(p1, p2, p3, p4)

                p1 = np.array(i_c[element[p1] - 1][0])
                p2 = np.array(i_c[element[p2] - 1][0])
                p3 = np.array(i_c[element[p3] - 1][0])
                p4 = np.array(i_c[element[p4] - 1][0])
                # print("p1:", p1)
                # print("p2:", p2)
                # print("p3:", p3)
                # print("p4:", p4)
                # print()

                v1 = np.linalg.norm(p1 - p2)
                v2 = np.linalg.norm(p3 - p4)
                length = 0.5 * (v1 + v2)
                A += 0.5 * length * d
                # print("A", A)
                # print()
        l0 = np.linalg.norm(np.array(i_c[node1][0]) - np.array(i_c[node2][0]))
        # A = np.sqrt(A)
        # A = A*d
        k = E * A / l0 #* (2 * r / l0)
        # print(k)
        e_p.append([k, l0, c])

    connections = np.array([connection for connection in connections if len(connection) == 4]) - 1
    return c_m, i_c, e_p, n, connections


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
    "g": 9.807,  # [m/s**2]   gravitational acceleration
    "v_w": [5, 0, 0],  # [m/s]     wind velocity vector
    'rho': 1.225,  # [kg/ m3]  air density
    'c_d_bridle': 1.05,  # [-]       drag-coefficient of bridles
    "d_bridle": 0.02  # [m]       diameter of bridle lines
}

# calculated parameters
params["l0"] = 0  # np.sqrt( 2 * (grid_length/(grid_size-1))**2)
params["m_segment"] = 1
params["k"] = E_t * d

# instantiate connectivity matrix and initial conditions array
c_matrix, init_cond, element_param, params["n"], element_list = cm_and_ic(filename, 1, E_t, params["c"])

# print(init_cond)

if __name__ == "__main__":
    print(element_param)

    import matplotlib.pyplot as plt

    x = []
    y = []
    z = []
    for i in range(len(init_cond)):
        x.append(init_cond[i][0][0])
        y.append(init_cond[i][0][1])
        z.append(init_cond[i][0][2])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    labels = ['Mesh particle', 'Mesh spring damper element']
    handles = []
    nodes = ax.scatter(x, y, z, c='red', label=labels[0])
    handles.append(nodes)
    for i, indices in enumerate(c_matrix):
        line = ax.plot([x[indices[0]], x[indices[1]]], [y[indices[0]], y[indices[1]]], [z[indices[0]], z[indices[1]]],
                       color='black', label=labels[1])
        if i == 0:
            handles.append(line[0])

    ax.legend(handles, labels)
    plt.title("Mesh Hencky problem")
    plt.show()
