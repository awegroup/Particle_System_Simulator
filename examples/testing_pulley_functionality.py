"""
Pulley test case replicating the kite_fem 7-node, 3-pulley topology in PSS.

kite_fem pulley element [A, P, B, k, c, l0]:
    Continuous rope A → around pulley node P → to B.
    Constraint:  l_AP + l_PB = l0   (total rope length is conserved).

PSS equivalent:
    Split each kite_fem pulley into two PULLEY-type SpringDamper links
    (A,P) and (P,B), cross-coupled via pulley_other_line_pair.
    Each link adds the OTHER segment's extension to its own:
        f_AP = -k * (l_AP - l0_AP + (l_PB - l0_PB)) * û_AP

Topology (7 nodes, 6 pulley links):
    Fixed:  0(0,0)  1(1,0)  2(2,0)  3(3,0)     ← top rail
    Free:   4(1,-1) 5(2,-1)                      ← pulley nodes
    Free:   6(1.5,-2)                             ← bottom payload

    Pulley ropes (kite_fem → PSS):
        [0,4,1] → link 0:(0,4) + link 1:(4,1)
        [2,5,3] → link 2:(2,5) + link 3:(5,3)
        [4,6,5] → link 4:(4,6) + link 5:(6,5)
"""

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
    """
    7-node cascading pulley network matching kite_fem topology.

    Three continuous ropes, each modelled as two cross-coupled PULLEY links:
        Rope 1: 0 → 4 → 1   (nodes 0,1 fixed on top rail)
        Rope 2: 2 → 5 → 3   (nodes 2,3 fixed on top rail)
        Rope 3: 4 → 6 → 5   (pulley nodes 4,5 feed into bottom node 6)
    """
    initial_conditions = [
        [[0.0, 0.0, 0.0], [0, 0, 0], 1.0, True],  # 0 — fixed
        [[1.0, 0.0, 0.0], [0, 0, 0], 1.0, True],  # 1 — fixed
        [[2.0, 0.0, 0.0], [0, 0, 0], 1.0, True],  # 2 — fixed
        [[3.0, 0.0, 0.0], [0, 0, 0], 1.0, True],  # 3 — fixed
        [[1.0, -1.0, 0.0], [0, 0, 0], 1.0, False],  # 4 — free (pulley point)
        [[2.0, -1.0, 0.0], [0, 0, 0], 1.0, False],  # 5 — free (pulley point)
        [[1.5, -2.0, 0.0], [0, 0, 0], 1.0, False],  # 6 — free (bottom payload)
    ]

    k_pulley = 1000.0
    c_pulley = 0.0

    # Six PULLEY links (two per rope)
    connectivity = [
        [0, 4, k_pulley, c_pulley, SpringDamperType.PULLEY],  # link 0  rope 1 seg A
        [4, 1, k_pulley, c_pulley, SpringDamperType.PULLEY],  # link 1  rope 1 seg B
        [2, 5, k_pulley, c_pulley, SpringDamperType.PULLEY],  # link 2  rope 2 seg A
        [5, 3, k_pulley, c_pulley, SpringDamperType.PULLEY],  # link 3  rope 2 seg B
        [4, 6, k_pulley, c_pulley, SpringDamperType.PULLEY],  # link 4  rope 3 seg A
        [6, 5, k_pulley, c_pulley, SpringDamperType.PULLEY],  # link 5  rope 3 seg B
    ]

    # Compute initial segment lengths (SpringDamper sets l0 = initial distance)
    pos = [np.array(ic[0]) for ic in initial_conditions]
    segments = [(0, 4), (4, 1), (2, 5), (5, 3), (4, 6), (6, 5)]
    rest = {i: np.linalg.norm(pos[a] - pos[b]) for i, (a, b) in enumerate(segments)}

    # Cross-coupling: each link watches the OTHER segment of its rope.
    # Format: "link_idx": [node_p3, node_p4, rest_length_p3p4]
    pulley_other_line_pair = {
        "0": [4, 1, rest[1]],  # link 0 (0→4)  watches link 1 (4→1)
        "1": [0, 4, rest[0]],  # link 1 (4→1)  watches link 0 (0→4)
        "2": [5, 3, rest[3]],  # link 2 (2→5)  watches link 3 (5→3)
        "3": [2, 5, rest[2]],  # link 3 (5→3)  watches link 2 (2→5)
        "4": [6, 5, rest[5]],  # link 4 (4→6)  watches link 5 (6→5)
        "5": [4, 6, rest[4]],  # link 5 (6→5)  watches link 4 (4→6)
    }

    params = {
        "dt": 0.01,
        "t_steps": 6000,
        "abs_tol": 1e-12,
        "rel_tol": 1e-8,
        "max_iter": 5000,
        "pulley_other_line_pair": pulley_other_line_pair,
    }

    ps = ParticleSystem(connectivity, initial_conditions, params, init_surface=False)
    return ps, connectivity, initial_conditions, params


def make_external_force(ps):
    """Force on the bottom payload node (6), matching kite_fem test."""
    f_ext = np.zeros(ps.n * 3)
    f_ext[6 * 3 + 0] = 50.0  # +X on node 6
    f_ext[6 * 3 + 1] = -100.0  # -Y on node 6
    return f_ext


def run_simulation(ps, params, f_ext):
    t_vector = np.linspace(
        params["dt"], params["t_steps"] * params["dt"], params["t_steps"]
    )
    e_kin = []
    f_int_norm = []
    final_step = t_vector[-1]

    for step in t_vector:
        ps.kin_damp_sim(f_ext)

        _, v = ps.x_v_current
        e_kin.append(np.linalg.norm(v * v))
        f_int_norm.append(np.linalg.norm(ps.f))
        final_step = step

        if len(e_kin) > 50 and np.mean(e_kin[-50:]) < 1e-12:
            break
        if np.isnan(e_kin[-1]) or np.isnan(f_int_norm[-1]):
            raise RuntimeError(f"Simulation diverged at t = {step:.4f}s")

    return (
        np.array(t_vector[: len(e_kin)]),
        np.array(e_kin),
        np.array(f_int_norm),
        final_step,
    )


def angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def verify_pulley_angles(ps, f_ext):
    """
    At a frictionless pulley node P the rope makes equal angles on both
    sides with respect to the continuation direction.

    For nodes 4 and 5 the continuation is the downstream rope to node 6.
    For node 6 (terminal pulley) there is no further rope — the external
    force direction acts as the reference.  Equal tension ⇒ the rope
    bisects the applied force, so ∠(4→6, f) == ∠(5→6, f).

    Pulley nodes and their ropes:
        node 4:  rope 1  →  0 → 4 → 1,   continuation 4→6
        node 5:  rope 2  →  2 → 5 → 3,   continuation 5→6  (6→5 reversed)
        node 6:  rope 3  →  4 → 6 → 5,   continuation = f_ext direction
    """
    x, _ = ps.x_v_current_3D
    force_vectors = f_ext.reshape(ps.n, 3)

    # Checks with a downstream node as reference
    node_checks = [
        # (label, pulley P, neighbour A, neighbour B, reference node)
        ("Node 4", 4, 0, 1, 6),
        ("Node 5", 5, 2, 3, 6),
    ]

    print("\nPulley angle verification (frictionless ⇒ equal angles):")
    all_ok = True

    for label, P, A, B, R in node_checks:
        vec_AP = x[P] - x[A]  # A → P
        vec_BP = x[P] - x[B]  # B → P
        vec_PR = x[R] - x[P]  # P → R  (downstream rope)

        angle_left = angle_between(vec_AP, vec_PR)
        angle_right = angle_between(vec_BP, vec_PR)
        diff = abs(angle_left - angle_right)
        ok = diff < 1.0
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        print(
            f"  {label}:  ∠({A}→{P}→{R}) = {angle_left:6.2f}°,  "
            f"∠({B}→{P}→{R}) = {angle_right:6.2f}°,  "
            f"Δ = {diff:.2f}°  [{status}]"
        )

    # Node 6: terminal pulley — use external force as reference direction
    P, A, B = 6, 4, 5
    vec_AP = x[P] - x[A]  # 4 → 6
    vec_BP = x[P] - x[B]  # 5 → 6
    f_dir = force_vectors[P, :3]  # external force on node 6

    angle_left = angle_between(vec_AP, f_dir)
    angle_right = angle_between(vec_BP, f_dir)
    diff = abs(angle_left - angle_right)
    ok = diff < 1.0
    status = "OK" if ok else "MISMATCH"
    if not ok:
        all_ok = False
    print(
        f"  Node 6:  ∠({A}→{P}→f) = {angle_left:6.2f}°,  "
        f"∠({B}→{P}→f) = {angle_right:6.2f}°,  "
        f"Δ = {diff:.2f}°  [{status}]"
    )

    return all_ok


# ── Plotting (kite_fem–style) ────────────────────────────────────────────

# Colour / style tables matching kite_fem Plotting.py defaults
_E_COLORS = {"spring": "red", "noncompressive": "blue", "pulley": "orange"}
_N_COLORS = {"fixed": "black", "free": "grey", "pulley": "red"}
_V_COLORS = {"external": "magenta", "internal": "cyan"}


def plot_state(
    ax,
    ps,
    connectivity,
    f_ext,
    title,
    pulley_nodes=None,
    fe_magnitude=0.35,
    plot_node_numbers=True,
):
    if pulley_nodes is None:
        pulley_nodes = set()

    x, _ = ps.x_v_current_3D
    n_nodes = len(x)

    # ── Nodes ──────────────────────────────────────────────────────────
    label_set = {"Fixed Node": False, "Free Node": False, "Pulley Node": False}
    for i in range(n_nodes):
        p = ps.particles[i]
        if i in pulley_nodes:
            ntype, col, sz = "Pulley Node", _N_COLORS["pulley"], 40
        elif p.fixed:
            ntype, col, sz = "Fixed Node", _N_COLORS["fixed"], 15
        else:
            ntype, col, sz = "Free Node", _N_COLORS["free"], 15

        ax.scatter(
            x[i, 0],
            x[i, 1],
            color=col,
            s=sz,
            zorder=20,
            edgecolors="black",
            linewidths=0.5,
            label=ntype if not label_set[ntype] else None,
        )
        label_set[ntype] = True

    # ── Node numbers ───────────────────────────────────────────────────
    if plot_node_numbers:
        for i, xi in enumerate(x):
            ax.text(
                xi[0] + 0.1,
                xi[1] + 0.1,
                str(i),
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7),
                zorder=80,
            )

    # ── Links (pulley ropes coloured orange, rest red) ─────────────────
    label_set_elem = {"Pulley spring": False, "Spring": False}
    for idx, (i, j, *rest) in enumerate(connectivity):
        ltype = rest[2] if len(rest) >= 3 else None
        if ltype == SpringDamperType.PULLEY:
            col, etype = _E_COLORS["pulley"], "Pulley spring"
        else:
            col, etype = _E_COLORS["spring"], "Spring"
        ax.plot(
            [x[i, 0], x[j, 0]],
            [x[i, 1], x[j, 1]],
            color=col,
            linewidth=1,
            zorder=3,
            label=etype if not label_set_elem.get(etype) else None,
        )
        label_set_elem[etype] = True

    # ── External force arrows (kite_fem–style quiver) ──────────────────
    force_vectors = f_ext.reshape(ps.n, 3)
    magnitudes = np.linalg.norm(force_vectors[:, :2], axis=1)
    max_mag = magnitudes.max() if magnitudes.max() > 0 else 1.0
    scale = fe_magnitude / max_mag  # arrow length per unit force

    for i in range(n_nodes):
        fx, fy = force_vectors[i, 0], force_vectors[i, 1]
        if fx != 0 or fy != 0:
            ax.quiver(
                x[i, 0],
                x[i, 1],
                fx,
                fy,
                angles="xy",
                scale_units="xy",
                scale=1.0 / scale if scale != 0 else 1.0,
                color=_V_COLORS["external"],
                linewidth=1,
                zorder=15,
            )
    # dummy for legend
    if magnitudes.max() > 0:
        ax.scatter(
            [],
            [],
            marker=r"$\longrightarrow$",
            c=_V_COLORS["external"],
            s=150,
            label="External Force Vector",
        )

    # ── Axis styling (kite_fem equal-range convention) ─────────────────
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # force minimum y to -2.5
    ylim = (min(ylim[0], -2.5), ylim[1])
    xmid = (xlim[0] + xlim[1]) / 2
    ymid = (ylim[0] + ylim[1]) / 2
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    ax.set_xlim(xmid - span / 2, xmid + span / 2)
    ax.set_ylim(ymid - span / 2, ymid + span / 2)
    ax.set_aspect("equal")
    ax.set(xlabel="x (m)", ylabel="y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def print_rope_lengths(ps, connectivity, initial_conditions):
    """Print per-rope total length vs initial total length."""
    x, _ = ps.x_v_current_3D
    pos0 = [np.array(ic[0]) for ic in initial_conditions]

    ropes = [
        ("Rope 1 (0→4→1)", [(0, 4), (4, 1)]),
        ("Rope 2 (2→5→3)", [(2, 5), (5, 3)]),
        ("Rope 3 (4→6→5)", [(4, 6), (6, 5)]),
    ]
    for label, segs in ropes:
        l_init = sum(np.linalg.norm(pos0[a] - pos0[b]) for a, b in segs)
        l_final = sum(np.linalg.norm(x[a] - x[b]) for a, b in segs)
        print(
            f"  {label}:  l0 = {l_init:.4f},  l_final = {l_final:.4f},  "
            f"Δ = {l_final - l_init:+.4f}"
        )


def main():
    ps, connectivity, initial_conditions, params = build_pulley_system()
    f_ext = make_external_force(ps)
    pulley_nodes = {4, 5, 6}  # nodes that act as frictionless pulleys

    # --- Initial state ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    plot_state(ax1, ps, connectivity, f_ext, "Initial Configuration", pulley_nodes)
    ax1.legend()

    # --- Run simulation ---
    t, e_kin, f_int, final_step = run_simulation(ps, params, f_ext)

    # --- Deformed state ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    plot_state(ax2, ps, connectivity, f_ext, "Deformed Configuration", pulley_nodes)
    ax2.legend()

    # --- Convergence history ---
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(t, e_kin, label="Kinetic energy proxy")
    ax3.plot(t, f_int, label="||f_int||")
    ax3.set_yscale("log")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Quantity")
    ax3.set_title("Convergence History")
    ax3.grid(alpha=0.3)
    ax3.legend()

    # --- Summary ---
    print(f"\nPulley example finished at t = {final_step:.3f}s")
    print("Final rope lengths:")
    print_rope_lengths(ps, connectivity, initial_conditions)

    # --- Pulley angle verification ---
    angles_ok = verify_pulley_angles(ps, f_ext)

    x, _ = ps.x_v_current_3D
    print(f"\nFinal node positions:")
    for i, xi in enumerate(x):
        tag = "fixed" if ps.particles[i].fixed else "free"
        print(f"  node {i} ({tag}): [{xi[0]:+.4f}, {xi[1]:+.4f}, {xi[2]:+.4f}]")

    if angles_ok:
        print("\n✓ Pulley angle check PASSED — equal angles at all pulley nodes.")
    else:
        print(
            "\n✗ Pulley angle check FAILED — angles differ at one or more pulley nodes."
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
