import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

# sys.path.append(os.path.abspath(".."))
from PSS.particleSystem.ParticleSystem import ParticleSystem

# dictionary of required parameters
params = {
    # model parameters
    "c": 100,  # [N s/m] damping coefficient
    "m_segment": 4,  # [kg] mass of each node
    # simulation settings
    "dt": 0.05,  # [s]       simulation timestep
    "t_steps": 1000,  # [-]      number of simulated time steps
    "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
    "rel_tol": 1e-5,  # [-]       relative error tolerance iterative solver
    "max_iter": 1e5,  # [-]       maximum number of iterations
    # physical parameters
    # "g": 9.807,  # [m/s^2]   gravitational acceleration
}




P = 5 # [N] force applied at the end of the cantilever 
P_Angle = 20   # [deg] angle of the force, 0 deg is in the y direction, 90 deg is in the z direction

P_y = P*np.cos(np.deg2rad(P_Angle)) # [N] force in y direction
P_z = P*np.sin(np.deg2rad(P_Angle)) # [N] force in z direction


EI = 100 # [N*m^2] bending stiffness of the cantilever


L = 6 # [m] length of the cantilever
L_side = 1 # [m] length of the square side



K_ei = (1/L_side)**3 # factor

print("PL^2/EI", (P*L**2/EI))


params["k"] = (EI*K_ei)/2.4# [N/m] spring stiffness
params["L"] = L # [m] length of the cantilever
params["n_row"] = int(L/L_side)+1 # Number of nodes per row 
params["k_cross"] = params["k"] # [N/m]  spring stiffness x crosses
params["n"] = params["n_row"]*4  # Number of nodes in the cantilever
params["L_0"] = params["L"] / (params["n_row"] - 1)  # [m] length of each segment

initial_conditions = []
connections = []

xyz_coordinates = np.empty((params["n"], 3))

for i in range(params["n"]):
    if i < params["n_row"]: 
        xyz_coordinates[i] = [i*params["L_0"],0, 0]
    elif i >= params["n_row"] and i < 2*params["n_row"]:
        xyz_coordinates[i] = [(i-params["n_row"])*params["L_0"],-L_side, 0]
    elif i >= 2*params["n_row"] and i < 3*params["n_row"]:
        xyz_coordinates[i] = [(i-2*params["n_row"])*params["L_0"],-L_side, L_side]
    else:
        xyz_coordinates[i] = [(i-3*params["n_row"])*params["L_0"],0, L_side]
        
    if i == 0 or i == params["n_row"] or i == 2*params["n_row"] or i == 3*params["n_row"]:
        initial_conditions.append([xyz_coordinates[i], np.zeros(3), params["m_segment"], True, [0,0,0],"point"])
    else:
        initial_conditions.append([xyz_coordinates[i], np.zeros(3), params["m_segment"], False, [0,0,0],"point"])

for i in range(params["n"]):
    if i < params["n_row"]-1:
        connections.append([i,i+1,params["k"], params["c"]])
        connections.append([i,i+params["n_row"],params["k"], params["c"]])
        connections.append([i,i+3*params["n_row"],params["k"], params["c"]])
        connections.append([i,i+params["n_row"]+1,params["k_cross"], params["c"]])
        connections.append([i,i+3*params["n_row"]+1,params["k_cross"], params["c"]])
        connections.append([i,i+2*params["n_row"],params["k_cross"], params["c"]])
    elif 2*params["n_row"]-1 > i > params["n_row"]-1 or 3*params["n_row"]-1 > i > 2*params["n_row"]-1:
        connections.append([i,i+1,params["k"], params["c"]])
        connections.append([i,i+params["n_row"],params["k"], params["c"]])
        connections.append([i,i+params["n_row"]+1,params["k_cross"], params["c"]])
        connections.append([i,i-params["n_row"]+1,params["k_cross"], params["c"]])
    elif 4*params["n_row"]-1 > i > 3*params["n_row"]-1:
        connections.append([i,i+1,params["k"], params["c"]])
        connections.append([i,i-3*params["n_row"]+1,params["k_cross"], params["c"]])
        connections.append([i,i-params["n_row"]+1,params["k_cross"], params["c"]])
        connections.append([i,i-2*params["n_row"],params["k_cross"], params["c"]])
    elif i == params["n_row"]-1:
        connections.append([i,i+3*params["n_row"],params["k"], params["c"]])
        connections.append([i,i+2*params["n_row"],params["k_cross"], params["c"]])
    elif i == 4*params["n_row"]-1:
        connections.append([i,i-params["n_row"],params["k"], params["c"]])
        connections.append([i,i-2*params["n_row"],params["k_cross"], params["c"]])
    else:
        connections.append([i,i-params["n_row"],params["k"], params["c"]])


        
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

f_ext = 0
f_ext = np.empty((params["n"], 3))
f_ext[params["n_row"]-1] = [0, -P_y/4, P_z/4]   # Apply a downward force on the last node
f_ext[2*params["n_row"]-1] = [0, -P_y/4,P_z/4]  # Apply a downward force on the last node
f_ext[3*params["n_row"]-1] = [0, -P_y/4, P_z/4]  # Apply a downward force on the last node
f_ext[4*params["n_row"]-1] = [0, -P_y/4, P_z/4]  # Apply a downward force on the last node


f_ext = f_ext.flatten()
        
# Plot the nodes
for i, node in enumerate(initial_conditions):
    if node[3]:  # Fixed node

        ax.scatter(node[0][0], node[0][1], node[0][2], color="red", marker="o", label="Fixed Node" if i == 0 else "")
        
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], node[0][2], color="blue", marker="o", label="Free Node" if i == 1 else "")

    # Plot the external forces as arrows
    ax.quiver(
        *node[0].tolist(), f_ext[3 * i], f_ext[3 * i + 1], f_ext[3 * i + 2], length=1
    )

for connection in connections:
    line = np.column_stack(
        [initial_conditions[connection[0]][0], initial_conditions[connection[1]][0]]
    )

    ax.plot(line[0], line[1], line[2], color="black")

ax.set_xlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_ylim(-params["L_0"]-params["L"]-.1,0.1)
ax.set_zlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Now we can setup the particle system and simulation
PS = ParticleSystem(connections, initial_conditions, params,init_surface=False)

t_vector = np.linspace(
    params["dt"], params["t_steps"] * params["dt"], params["t_steps"]
)
final_step = 0
E_kin = []
f_int = []

f_external = f_ext
# print(
#     f"f_external: shape: {np.shape(f_external)}, average: {np.mean(f_external)}, min: {np.min(f_external)}, max: {np.max(f_external)}"
# )

# And run the simulation

for step in t_vector:
    PS.kin_damp_sim(f_ext)

    final_step = step
    (
        x,
        v,
    ) = PS.x_v_current
    E_kin.append(np.linalg.norm(v * v))
    f_int.append(np.linalg.norm(PS.f_int))

    converged = False
    if step > 10:
        if np.max(E_kin[-10:-1]) <= 1e-29:
            converged = True
    if converged and step > 1:
        print("Kinetic damping PS converged", step)
        break

particles = PS.particles

final_positions = [
    [particle.x, particle.v, particle.m, particle.fixed, particle.constraint_type] for particle in PS.particles
]

# Plotting final results in 2D
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Plot the nodes
for i, node in enumerate(final_positions):
    if node[3]:  # Fixed node
        ax.scatter(node[0][0], node[0][1], node[0][2], color="red", marker="o", label="Fixed Node" if i == 0 else "")
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], node[0][2], color="blue", marker="o", label="Free Node" if i == 1 else "")

    # Plot the external forces as arrows
    ax.quiver(
        *node[0].tolist(), f_ext[3 * i], f_ext[3 * i + 1], f_ext[3 * i + 2]
    )  # , length  = 0.3)

for connection in connections:
    line = np.column_stack(
        [final_positions[connection[0]][0], final_positions[connection[1]][0]]
    )
    ax.plot(line[0], line[1], line[2], color="black")


# Set axis limits
ax.set_xlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_ylim(-params["L_0"]-params["L"]-.1,0.1)

# Add labels and title
ax.set_xlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_ylim(-params["L_0"]-params["L"]-.1,0.1)
ax.set_zlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection="3d")


# Plot the nodes
for i, node in enumerate(initial_conditions):
    if node[3]:  # Fixed node
        ax.scatter(node[0][0], node[0][1], node[0][2], color="red", marker="o")
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], node[0][2], color="blue", marker="o")
        
for i, node in enumerate(final_positions):
    if node[3]:  # Fixed node
        ax.scatter(node[0][0], node[0][1], node[0][2], color="red", marker="o")
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], node[0][2], color="blue", marker="o")
        

# Plot the connections between nodes
for connection in connections:
    line = np.column_stack(
        [initial_conditions[connection[0]][0], initial_conditions[connection[1]][0]]
    )

    ax.plot(line[0], line[1], line[2], color="black")
    line = np.column_stack(
        [final_positions[connection[0]][0], final_positions[connection[1]][0]]
    )
    ax.plot(line[0], line[1], line[2], color="red")
    


x_init = []
y_init = []
z_init = []
x_final = []
y_final = []
z_final = []
for i in range(params["n_row"]):
    x_init.append((initial_conditions[i][0][0] + initial_conditions[i+2*params["n_row"]][0][0]) / 2)
    y_init.append((initial_conditions[i][0][1] + initial_conditions[i+2*params["n_row"]][0][1]) / 2)
    z_init.append((initial_conditions[i][0][2] + initial_conditions[i+2*params["n_row"]][0][2]) / 2)
    x_final.append((final_positions[i][0][0] + final_positions[i+2*params["n_row"]][0][0]) / 2)
    y_final.append((final_positions[i][0][1] + final_positions[i+2*params["n_row"]][0][1]) / 2)
    z_final.append((final_positions[i][0][2] + final_positions[i+2*params["n_row"]][0][2]) / 2)
    
plt.plot(x_init, y_init, z_init, "o-", label="Initial state")
plt.plot(x_final, y_final, z_final, "o-", label="Final state")
ax.set_xlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_ylim(-params["L_0"]-params["L"]-.1,0.1)

# Add labels and title
ax.set_xlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_ylim(-params["L_0"]-params["L"]-.1,0.1)
ax.set_zlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()



angle = np.arctan2(((y_final[-1] - y_final[-2])**2+ (z_final[-1] - z_final[-2])**2)**0.5 , x_final[-1] - x_final[-2])
print("w",(((y_final[-1]**2+z_final[-1]**2)**0.5)-((y_init[-1]**2+z_init[-1]**2)**0.5))/L)
print("u",(x_final[-1] - x_init[-1])/-L)
print("angle",angle)