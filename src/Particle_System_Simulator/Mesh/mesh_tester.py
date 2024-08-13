# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:44:09 2024

@author: Mark Kalsbeek
"""
import time
import logging

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from Particle_System_Simulator.particleSystem.ParticleSystem import ParticleSystem
from Particle_System_Simulator.Sim.simulations import Simulate_1d_Stretch
import Particle_System_Simulator.Mesh.mesh_functions as MF


class AnalyseMaterial:
    def __init__(self):
        self.params = {
            # model parameters
            "k": 1,  # [N/m]   spring stiffness
            "k_d": 1,  # [N/m] spring stiffness for diagonal elements
            "c": 1,  # [N s/m] damping coefficient
            "m_segment": 1,  # [kg] mass of each node
            # simulation settings
            "dt": 0.1,  # [s]       simulation timestep
            "t_steps": 1000,  # [-]      number of simulated time steps
            "abs_tol": 1e-50,  # [m/s]     absolute error tolerance iterative solver
            "rel_tol": 1e-9,  # [-]       relative error tolerance iterative solver
            "max_iter": 1e2,  # [-]       maximum number of iterations]
            # Simulation Steps
            "steps": np.linspace(0.001, 0.1, 10),  # [-] Strain steps
            # Mesh_dependent_settings
            "midstrip_width": 0.1,
            "boundary_margin": 0.15,
        }
        self.history = {}

    def stiffness_sweep(self, chirality=False):
        # performs tests for range of ratios of k to k_d

        if chirality:
            raise NotImplementedError

        stiffness_ratio_range = np.linspace(0.3, 0.5, 9)
        total_cases = len(stiffness_ratio_range) * len(self.params["steps"])
        msg = f"Starting Stiffness sweep. Analysing {total_cases} in total."
        print("=" * len(msg))
        print(msg)

        start_time = time.time()
        last_time = time.time()
        # We will keep k constant and alter k_d
        testdata = []
        for i, ratio in enumerate(stiffness_ratio_range):
            self.params["k_d"] = self.params["k"] * ratio
            result = self.run_testcase(ratio)
            curr_time = time.time()
            dt = curr_time - last_time
            t_rest = (len(stiffness_ratio_range) - i) * dt
            print(
                f"That took {dt:.1f}s, est. time remaining {t_rest//60:.0f}m {t_rest%60:.1f}s"
            )

            last_time = curr_time
            for item in result.items():
                testdata.append((ratio, item[0], item[1][0], item[1][1]))

        dtype = [
            ("Stiffness_ratio", "f4"),
            ("Strain", "f4"),
            ("Reaction", "f4"),
            ("Poisson", "f4"),
        ]
        shape = (len(stiffness_ratio_range), len(self.params["steps"]))
        self.result = np.array(testdata, dtype=dtype).reshape(shape)

    def run_testcase(self, case_id):
        params = self.params

        n_segments = 20
        initial_conditions, connections = MF.mesh_square_cross(
            1, 1, 1 / n_segments, params
        )
        # initial_conditions, connections = MF.mesh_rotate_and_trim(initial_conditions,
        #                                                        connections,
        #                                                        45/2)
        PS = ParticleSystem(connections, initial_conditions, params)

        Sim = Simulate_1d_Stretch(PS, params)

        msg = f"\nStarting simulation batch for {params['k']=}{params['k_d']=}\n"
        print("-" * len(msg))
        print(msg)

        Sim.run_simulation()

        for key in Sim.history.keys():
            reaction, poisson = Sim.history[key]
            print(f"Strain {key}, {reaction=}, {poisson=}")

        print("=" * len(msg))
        print("\n\n")

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        PS.plot(ax, colors="strain")
        strain = params["steps"][-1]
        title = f"MaterialTest, {params['k']=:.4f},{params['k_d']=:.4f},{strain=:.4f}".strip(
            "params"
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(f"temp\{title}.jpg", dpi=200, format="jpg")
        plt.close(fig)

        return Sim.history

    def plot_results(self):
        strain = self.result["Strain"].T
        poisson = self.result["Poisson"].T
        labels = self.result["Stiffness_ratio"][:, 0]
        labels = [f"k_d/k: {i}" for i in labels]
        reaction_force = self.result["Reaction"].T

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(strain, reaction_force, label=labels)
        ax1.set_title("Reaction Force versus Strain")
        ax1.set_xlabel = "strain"
        ax1.set_ylabel = "Reaction Force"
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.plot(strain, poisson, label=labels)
        ax2.set_title("Poissons Ratio versus Strain")
        ax2.set_xlabel = "strain"
        ax2.set_ylabel = "Poissons Ratio"
        ax2.legend()


if __name__ == "__main__":
    AM = AnalyseMaterial()
    AM.stiffness_sweep()
