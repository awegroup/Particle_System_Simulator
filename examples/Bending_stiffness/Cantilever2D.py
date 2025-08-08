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
EI = 400 # [N*m^2] bending stiffness of the cantilever
k_cross = EI # [N/m] other spring stiffness

L = 10 # [m] length of the cantilever
L_side = 1 # [m] length of the square side



K_ei = (1/L_side)**3 # factor

print("PL^2/EI", (P*L**2/EI))


params["k"] = EI*K_ei # [N/m] spring stiffness
params["L"] = L # [m] length of the cantilever
params["n_row"] = int(L/L_side)+1 # Number of nodes per row 
params["k_cross"] = k_cross # [N/m]  spring stiffness x crosses
params["n"] = params["n_row"]*2  # Number of nodes in the cantilever
params["L_0"] = params["L"] / (params["n_row"] - 1)  # [m] length of each segment

initial_conditions = []
connections = []

xyz_coordinates = np.empty((params["n"], 3))
for i in range(params["n"]):
    if i < params["n_row"]: 
        xyz_coordinates[i] = [i*params["L_0"],0, 0]
    else:
        xyz_coordinates[i] = [(i-params["n_row"])*params["L_0"],-L_side, 0]
    if i == 0 or i == params["n_row"]:
        initial_conditions.append([xyz_coordinates[i], np.zeros(3), params["m_segment"], True, [0,0,0],"point"])
    else:
        initial_conditions.append([xyz_coordinates[i], np.zeros(3), params["m_segment"], False, [0,0,0],"point"])
    
for i in range(params["n"]-1):
    if i < params["n_row"]-1:
        connections.append([i,i+1,params["k"], params["c"]])
        connections.append([i,i+params["n_row"],params["k"], params["c"]])
        connections.append([i,i+params["n_row"]+1,params["k_cross"], params["c"]])
    elif i == params["n_row"]-1:
        connections.append([i,i+params["n_row"],params["k"], params["c"]])
    else:
        connections.append([i,i+1,params["k"], params["c"]])
        connections.append([i,i-params["n_row"]+1,params["k_cross"], params["c"]])

fig, ax = plt.subplots(figsize=(6,6))

f_ext = np.zeros((params["n"], 3))
f_ext[params["n_row"]-1] = [0, -P/2, 0]   # Apply a downward force on the last node
f_ext[-1] = [0, -P/2, 0]  # Apply a downward force on the last node
f_ext = f_ext.flatten()
print(f_ext)
        
# Plot the nodes
for i, node in enumerate(initial_conditions):
    if node[3]:  # Fixed node

        ax.scatter(node[0][0], node[0][1], color="red", marker="o", label="Fixed Node" if i == 0 else "")
        
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], color="blue", marker="o", label="Free Node" if i == 1 else "")

    # Plot the external forces as arrows
    ax.quiver(
        node[0][0], node[0][1], f_ext[3 * i], f_ext[3 * i + 1], angles="xy", scale_units="xy", scale=1, color="green"
    )

# Plot the connections between nodes
for i,connection in enumerate(connections):
    line = np.column_stack(
        [initial_conditions[connection[0]][0][:2], initial_conditions[connection[1]][0][:2]]
    )

    ax.plot(line[0], line[1], color="black")

ax.set_xlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_ylim(-params["L_0"]-params["L"]-.1,0.1)
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
fig, ax = plt.subplots(figsize=(6,6))

# Plot the nodes
for i, node in enumerate(final_positions):
    if node[3]:  # Fixed node
        ax.scatter(node[0][0], node[0][1], color="red", marker="o", label="Fixed Node" if i == 0 else "")
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], color="blue", marker="o", label="Free Node" if i == 1 else "")

    # Plot the external forces as arrows
    ax.quiver(
        node[0][0], node[0][1], f_ext[3 * i], f_ext[3 * i + 1], angles="xy", scale_units="xy", scale=1, color="green"
        )

# Plot the connections between nodes
for i,connection in enumerate(connections):
    line = np.column_stack(
        [final_positions[connection[0]][0][:2], final_positions[connection[1]][0][:2]]
    )
    ax.plot(line[0], line[1], color="black")

# Set axis limits
ax.set_xlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_ylim(-params["L_0"]-params["L"]-.1,0.1)

# Add labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Final State")

# Add legend
ax.legend()

# Show the plot
plt.show()


fig, ax = plt.subplots(figsize=(6,6))

# Plot the nodes
for i, node in enumerate(initial_conditions):
    if node[3]:  # Fixed node
        ax.scatter(node[0][0], node[0][1], color="red", marker="o")
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], color="blue", marker="o")
        
for i, node in enumerate(final_positions):
    if node[3]:  # Fixed node
        ax.scatter(node[0][0], node[0][1], color="red", marker="o")
    else:  # Free node
        ax.scatter(node[0][0], node[0][1], color="blue", marker="o")
        

# Plot the connections between nodes
for i, connection in enumerate(connections):
    line = np.column_stack(
        [initial_conditions[connection[0]][0][:2], initial_conditions[connection[1]][0][:2]]
    )
    ax.plot(line[0], line[1], color="black",label="Initial state" if i == 0 else "")
    line = np.column_stack(
        [final_positions[connection[0]][0][:2], final_positions[connection[1]][0][:2]]
    )
    ax.plot(line[0], line[1], color="red",label="Final state" if i == 0 else "")
    
ax.set_xlim(-.1, params["L"]+params["L_0"]+.1)
ax.set_ylim(-params["L_0"]-params["L"]-.1,0.1)    




x_init = []
y_init = []
x_final = []
y_final = []
for i in range(params["n_row"]):
    x_init.append((initial_conditions[i][0][0] + initial_conditions[i+params["n_row"]][0][0]) / 2)
    y_init.append((initial_conditions[i][0][1] + initial_conditions[i+params["n_row"]][0][1]) / 2)
    x_final.append((final_positions[i][0][0] + final_positions[i+params["n_row"]][0][0]) / 2)
    y_final.append((final_positions[i][0][1] + final_positions[i+params["n_row"]][0][1]) / 2)

plt.plot(x_init, y_init, "o-", label="Initial state")
plt.plot(x_final, y_final, "o-", label="Final state")

plt.legend()
plt.show()


angle = np.arctan2(y_final[-1] - y_final[-2], x_final[-1] - x_final[-2])

print("w",(y_final[-1] - y_init[-1])/-L)
print("u",(x_final[-1] - x_init[-1])/-L)
print("angle",angle)