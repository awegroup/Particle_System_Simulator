import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

try:
    from PSS.particleSystem.ParticleSystem import ParticleSystem
    from PSS.particleSystem.SpringDamper import SpringDamperType
except ModuleNotFoundError:
    root_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isfile(os.path.join(root_dir, ".gitignore")):
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))
        if root_dir == "/":
            raise FileNotFoundError("Could not find repository root directory.")
    sys.path.insert(0, str(Path(root_dir) / "src"))
    from PSS.particleSystem.ParticleSystem import ParticleSystem
    from PSS.particleSystem.SpringDamper import SpringDamperType


def build_pulley_system():
    initial_conditions = [
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, True],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, True],
        [[-1.0, -1.0, 0.0], [0.0, 0.0, 0.0], 1.0, True, [0.0, 0.0, 1.0], "plane"],
        [[1.0, -1.0, 0.0], [0.0, 0.0, 0.0], 1.0, True, [0.0, 0.0, 1.0], "plane"],
    ]

    # Two pulley links that are coupled to each other's line length.
    connectivity = [
        [0, 2, 200.0, 2.0, SpringDamperType.PULLEY],
        [1, 3, 200.0, 2.0, SpringDamperType.PULLEY],
        [2, 3, 40.0, 1.0, SpringDamperType.DEFAULT],
    ]

    rest_0_2 = np.linalg.norm(
        np.array(initial_conditions[0][0]) - np.array(initial_conditions[2][0])
    )
    rest_1_3 = np.linalg.norm(
        np.array(initial_conditions[1][0]) - np.array(initial_conditions[3][0])
    )

    params = {
        "dt": 0.01,
        "t_steps": 4000,
        "abs_tol": 1e-12,
        "rel_tol": 1e-8,
        "max_iter": 5000,
        "pulley_other_line_pair": {
            "0": [1, 3, rest_1_3],  # pulley link 0 is coupled to line (1, 3)
            "1": [0, 2, rest_0_2],  # pulley link 1 is coupled to line (0, 2)
        },
    }

    ps = ParticleSystem(connectivity, initial_conditions, params, init_surface=False)
    return ps, connectivity, params


def make_external_force(ps):
    f_ext = np.zeros(ps.n * 3)
    f_ext[2 * 3 + 1] = -25.0
    f_ext[3 * 3 + 1] = -25.0
    return f_ext


def run_simulation(ps, params, f_ext):
    t_vector = np.linspace(
        params["dt"], params["t_steps"] * params["dt"], params["t_steps"]
    )
    e_kin = []
    f_int = []
    final_step = t_vector[-1]

    for step in t_vector:
        ps.kin_damp_sim(f_ext)

        _, v = ps.x_v_current
        e_kin.append(np.linalg.norm(v * v))
        f_int.append(np.linalg.norm(ps.f))
        final_step = step

        if len(e_kin) > 50 and np.mean(e_kin[-50:]) < 1e-12:
            break
        if np.isnan(e_kin[-1]) or np.isnan(f_int[-1]):
            raise RuntimeError(f"Simulation diverged at t = {step:.4f}s")

    return np.array(t_vector[: len(e_kin)]), np.array(e_kin), np.array(f_int), final_step


def plot_state(ax, ps, connectivity, f_ext, title):
    x, _ = ps.x_v_current_3D
    fixed_mask = np.array(
        [p.fixed and p.constraint_type == "point" for p in ps.particles], dtype=bool
    )
    free_mask = ~fixed_mask

    ax.scatter(
        x[fixed_mask, 0], x[fixed_mask, 1], color="red", marker="o", label="Fixed nodes"
    )
    ax.scatter(
        x[free_mask, 0], x[free_mask, 1], color="blue", marker="o", label="Free nodes"
    )

    for i, j, *_ in connectivity:
        ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], color="black", linewidth=1)

    force_vectors = f_ext.reshape(ps.n, 3)
    ax.quiver(
        x[:, 0],
        x[:, 1],
        force_vectors[:, 0],
        force_vectors[:, 1],
        angles="xy",
        scale_units="xy",
        scale=70,
        color="tab:green",
        label="External force",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main():
    ps, connectivity, params = build_pulley_system()
    f_ext = make_external_force(ps)

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_state(ax1, ps, connectivity, f_ext, "Initial Configuration")
    ax1.legend()

    t, e_kin, f_int, final_step = run_simulation(ps, params, f_ext)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_state(ax2, ps, connectivity, f_ext, "Deformed Configuration")
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(t, e_kin, label="Kinetic energy proxy")
    ax3.plot(t, f_int, label="||f_int||")
    ax3.set_yscale("log")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Quantity")
    ax3.set_title("Convergence History")
    ax3.grid(alpha=0.3)
    ax3.legend()

    print(f"Pulley example finished at t = {final_step:.3f}s")
    plt.show()


if __name__ == "__main__":
    main()
