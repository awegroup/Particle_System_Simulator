"""
Script for PS framework validation, benchmark case where tether is fixed at both ends and is deflected by perpendicular
wind flow
"""
import numpy as np
import numpy.typing as npt
import tether_deflection_windFlow_input as input
import matplotlib.pyplot as plt
import pandas as pd
import sys
from Msc_Alexander_Batchelor.src.particleSystem.ParticleSystem import ParticleSystem

from sympy import *
def instantiate_ps():
    return ParticleSystem(input.c_matrix, input.init_cond, input.params)


def generate_animation(pos, n: int, t: npt.ArrayLike):
        from matplotlib import animation
        import matplotlib
        import math
        matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\FFmpeg\\bin\\ffmpeg.exe'
        filename = f"windFlow_deflection-{input.params['n']}Particles-{input.params['k']}stiffness-{input.params['c']}"\
                   f"damping_coefficient-{input.params['dt']}timestep-{input.params['t_steps']}steps-.mov"
        savelocation = r"C:\\Users\\Alexander\\Documents\\Master\\Thesis\\Figures\\GIFs\\"

        # configuration of plot
        fig, ax = plt.subplots()
        ax.set_xlim((-1, 0.5))
        ax.set_ylim((-1, 11))
        ax.grid(which='major')
        plt.ylabel("height [m]")
        plt.xlabel("x position [m]")
        plt.title(f"Animation of tether deflected by perpendicular wind flow")

        # calculation which values for each frame
        fps = 60  # 1 / input.params['dt']
        multi = round(input.params['dt']**-1 / fps)
        n_frames = math.floor(len(t)/multi)
        frame_indeces = [i * multi for i in range(n_frames)]

        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i):
            index = frame_indeces[i]
            timestep = t[index]
            x = pos.loc[timestep, [f'x{j + 1}' for j in range(n)]]
            y = pos.loc[timestep, [f'z{j + 1}' for j in range(n)]]
            line.set_data(x, y)
            return (line,)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=20, blit=True)  # , save_count=len(self.t))

        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(savelocation + filename, writer=writervideo)
        plt.cla()
        return


def calculate_f_a(ps: ParticleSystem):
    particle_list = ps.particles
    f_a = np.zeros(input.params['n']*3, )
    rho = 1.225

    for i in range(len(particle_list) - 1):
        V_b = 0.5 * (particle_list[i].v + particle_list[i + 1].v)  # velocity of the bridle = avg vel. of the particles
        V_b_app = input.params["v_w"] - V_b  # apparent velocity of bridle
        V_b_norm = np.linalg.norm(V_b_app)
        x = particle_list[i].x - particle_list[i + 1].x
        l_element = np.linalg.norm(x)

        # derivation of equation below, see "Bridle Particle pdf"
        S_eff_bridle = input.params["d_bridle"] * np.sqrt(l_element ** 2 - (np.dot(V_b_app, x) / V_b_norm) ** 2)
        F_a_drag = 0.5 * rho * V_b_app * V_b_norm * S_eff_bridle * input.params['c_d_bridle']
        # Drag force, includes the direction of the velocity
        f_a[i * 3:(i + 1) * 3] += 0.5 * F_a_drag
        f_a[(i + 1) * 3:(i + 2) * 3] += 0.5 * F_a_drag

    return f_a


def exact_solution():
    # analytical steady state solution for particles position
    import sympy as sp

    k = input.params["k"]
    n = input.params["n"]
    cd = input.params["c_d_bridle"]
    d = input.params["d_bridle"]
    rho = input.params["rho"]
    vw = input.params["v_w"]
    l0 = input.params["l0"]
    l = input.params["L"]

    ux1, uy1 = sp.symbols("ux1 uy1")
    ux2, uy2 = sp.symbols("ux2 uy2")
    ux3, uy3 = sp.symbols("ux3 uy3")

    f = 0.5*0.5*rho*cd*(1-uy1)*d*np.linalg.norm(vw)**2 + 0.5*0.5*rho*cd*(1+uy1)*d*np.linalg.norm(vw)**2

    F = sp.Matrix([0, 0, f, 0, 0, 0])
    U = sp.Matrix([ux1, uy1, ux2, uy2, ux3, uy3])

    t1 = sp.atan((1 + uy2) / ux2)
    t2 = pi - sp.atan((1 - uy2) / ux2)

    # t1, t2 = sp.symbols("t1 t2")

    K1 = k*sp.Matrix([[sp.cos(t1)*sp.cos(t1), sp.sin(t1)*sp.cos(t1), -sp.cos(t1)*sp.cos(t1), -sp.sin(t1)*sp.cos(t1)],
                      [sp.sin(t1)*sp.cos(t1), sp.sin(t1)*sp.sin(t1), -sp.sin(t1)*sp.cos(t1), -sp.sin(t1)*sp.sin(t1)],
                      [-sp.cos(t1)*sp.cos(t1), -sp.sin(t1)*sp.cos(t1), sp.cos(t1)*sp.cos(t1), sp.sin(t1)*sp.cos(t1)],
                      [-sp.sin(t1)*sp.cos(t1), -sp.sin(t1)*sp.sin(t1), sp.sin(t1)*sp.cos(t1), sp.sin(t1)*sp.sin(t1)]])

    K2 = k*sp.Matrix([[sp.cos(t2)*sp.cos(t2), sp.sin(t2)*sp.cos(t2), -sp.cos(t2)*sp.cos(t2), -sp.sin(t2)*sp.cos(t2)],
                      [sp.sin(t2)*sp.cos(t2), sp.sin(t2)*sp.sin(t2), -sp.sin(t2)*sp.cos(t2), -sp.sin(t2)*sp.sin(t2)],
                      [-sp.cos(t2)*sp.cos(t2), -sp.sin(t2)*sp.cos(t2), sp.cos(t2)*sp.cos(t2), sp.sin(t2)*sp.cos(t2)],
                      [-sp.sin(t2)*sp.cos(t2), -sp.sin(t2)*sp.sin(t2), sp.sin(t2)*sp.cos(t2), sp.sin(t2)*sp.sin(t2)]])

    K = sp.Matrix(zeros(n*2, n*2))      # 2d evaluation for now
    K[0:4, 0:4] += K1
    K[2:, 2:] += K2

    # K = K.col_insert(6, F)
    # print(K)
    U = sp.Matrix([ux2, uy2])
    F = sp.Matrix([f, 0])
    K = K[2:4, 2:4]
    soe = K*U - F
    x = (0.1, 0)

    u = (ux2, uy2)
    soe = (soe[0], soe[1])
    # print(soe)
    # return sp.solve_linear_system(K, U)#  sp.solve(soe, (ux2, uy2))

    return sp.solvers.solvers.nsolve(soe, u, x)


def plot(psystem: ParticleSystem):
    n = input.params['n']
    t_vector = np.linspace(input.params["dt"], input.params["t_steps"] * input.params["dt"], input.params["t_steps"])

    x = {}
    v = {}
    for i in range(n):
        x[f"x{i + 1}"] = np.zeros(len(t_vector))
        x[f"y{i + 1}"] = np.zeros(len(t_vector))
        x[f"z{i + 1}"] = np.zeros(len(t_vector))
        v[f"vx{i + 1}"] = np.zeros(len(t_vector))
        v[f"vy{i + 1}"] = np.zeros(len(t_vector))
        v[f"vz{i + 1}"] = np.zeros(len(t_vector))

    position = pd.DataFrame(index=t_vector, columns=x)
    velocity = pd.DataFrame(index=t_vector, columns=v)

    g = input.params["g"]
    n = input.params["n"]
    f_ext = np.array([[0, 0, 0] for i in range(n)]).flatten()

    for step in t_vector:           # propagating the simulation for each timestep and saving results
        f_aero = calculate_f_a(ps)
        position.loc[step], velocity.loc[step] = psystem.simulate(f_ext + f_aero)

    # generate animation of results, requires smarter configuration to make usable on other PCs
    # generate_animation(position, n, t_vector)

    # generating analytical solution for the same time vector
    # exact, decay = exact_solution(t_vector)

    # plotting & graph configuration
    for i in range(n):
        position[f"x{i + 1}"].plot()
    # plt.plot(t, exact)
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Validation PS framework, deflection of particles by wind flow, with Implicit Euler scheme")
    plt.legend([f"displacement particle {i + 1}" for i in range(n)])
    plt.grid()

    # saving resulting figure
    figure = plt.gcf()
    figure.set_size_inches(8.3, 5.8)  # set window to size of a3 paper

    # Not sure if this is the smartest way to automate saving results relative to other users directories
    file_path = sys.path[1] + "/Msc_Alexander_Batchelor/code_Validation/benchmark_results/" \
                              "tether_deflection_windFlow/"
    img_name = f"{input.params['n']}Particles-{input.params['k_t']}stiffness-{input.params['c']}damping_coefficient-" \
               f"{input.params['dt']}timestep-{input.params['t_steps']}steps.jpeg"
    plt.savefig(file_path + img_name, dpi=300, bbox_inches='tight')

    plt.show()

    return


if __name__ == "__main__":
    ps = instantiate_ps()

    plot(ps)

    print(exact_solution())
    #
    # stiffness = ps.stiffness_m
    # b = calculate_f_a(ps)
    # print(b)
    # print(stiffness)
    # print(np.linalg.det(stiffness))
    # stiffness = stiffness[stiffness != 0]
    # print(stiffness)
    # print(np.linalg.det([[-2000, 2000, 0], [2000, -4000, 2000], [0, 2000, -2000]]))
    #
    # from scipy.sparse.linalg import bicgstab
    # x, _ = bicgstab(stiffness, b)
    #
    #
    # print(x)
    # print(_)

