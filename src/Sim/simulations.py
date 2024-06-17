# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:21:49 2023

@author: Mark Kalsbeek

This module includes tools to aid in simulation, as well as some pre-baked simulation functions
"""


import time
import logging

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from ..particleSystem.ParticleSystem import ParticleSystem
from ..Mesh import mesh_functions as MF



class Simulate:
    def __init__(self, ParticleSystem):
        self.PS = ParticleSystem

    def run_simulation(self):
        pass

class Simulate_1d_Stretch(Simulate):
    """
    Simulation class that runs a 1D stretch to asses poisson ratio of mesh
    """
    def __init__(self, ParticleSystem, sim_params, save_plots = False):
        self.PS = ParticleSystem
        self.params = sim_params
        self.steps = sim_params['steps'] # represent strain values
        self.history = {}

    def run_simulation(self):
        x_cleaned = np.array([particle.x for particle in self.PS.particles])
        starting_dimentions = np.ptp(x_cleaned, axis = 0)

        midstrip_indices = MF.ps_find_mid_strip_y(self.PS,
                                                  self.params['midstrip_width'])

        self.PS, boundaries = MF.ps_fix_opposite_boundaries_x(self.PS,
                                                              margin = self.params['boundary_margin'])


        total_displacement = 0
        for strain in self.steps:
            starting_time = time.time()
            print(f'Starting simulation on step {strain=}')
            displacement = starting_dimentions[0] * strain - total_displacement
            total_displacement += displacement
            MF.ps_stretch_in_x(self.PS, boundaries[1], displacement)


            converged = False
            convergence_history = []
            while not converged:
                self.PS.kin_damp_sim()

                #convergence check
                ptp_range = MF.ps_find_strip_dimentions(self.PS, midstrip_indices)
                transverse_strain = (ptp_range[1]-starting_dimentions[1])/starting_dimentions[1]

                reaction_force = MF.ps_find_reaction_of_boundary(self.PS, boundaries[1])
                reaction_force = np.linalg.norm(reaction_force)

                e_kin = self.PS.kinetic_energy
                convergence_history.append(e_kin)
                step = len(convergence_history)
                if step>5:
                    crit = abs(convergence_history[-1]-convergence_history[-2])
                    if crit < 1e-15:
                        converged = True
                    if not step%50:
                        print(f'Just finished {step=}, {crit=:.2e}')
            finished_time = time.time()
            delta_time = finished_time - starting_time


            poissons_ratio = transverse_strain / strain

            self.history[strain] = [reaction_force, poissons_ratio]

            print(f'Finished with {transverse_strain=:.4f} and force {reaction_force=:.2f}')
            print(f'That took  {delta_time//60:.0f}m {delta_time%60:.2f}s')
            print('\n')


    def plot_results(self):
        reaction_force = []
        poissons_ratio = []

        for step in self.steps:
            force, ratio = self.history[step]
            reaction_force.append(force)
            poissons_ratio.append(ratio)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(self.steps, reaction_force)
        ax1.set_title("Reaction Force versus Strain")

        ax2 = fig.add_subplot(122)
        ax2.plot(self.steps, poissons_ratio)
        ax2.set_title("Poissons Ratio versus Strain")

class Simulate_1d_Shear(Simulate):
    """
    Simulation class that runs a 1D shear to asses accuracy of spring value assignment
    """

    def __init__(self, ParticleSystem: ParticleSystem):
        self.PS = ParticleSystem
        self.params = self.PS.params
        self.history = {}

        required = ["dt", "abs_tol", "rel_tol", "convergence_threshold"]

        for key in required:
            if not key in self.params.keys():
                raise KeyError(f"{key} missing from params")

        # Prepare the particle system
        x,_ = self.PS.x_v_current_3D
        peak_values = np.ptp(x, axis = 0)

        # Assign constraints and collect line-constrained particles to apply forces
        # Bottom row = fixed
        # Top row = line constrained to move only in x axis
        self.line_mask = np.zeros(len(self.PS.particles)*3)
        for i, particle in enumerate(self.PS.particles):
            if particle.x[1] == 0:
                particle.set_fixed(True)
            if particle.x[1] == peak_values[1]:
                particle.set_fixed(True,
                                   constraint = [1,0,0],
                                   constraint_type = 'line')
                self.line_mask[i*3] = True
        self.line_indices = np.nonzero(self.line_mask)[0]

    def run_simulation(self, force=1):
        starting_time = time.time()
        print('Starting simulation')
        forces = self.line_mask * force
        self.history['step'] = 0
        self.history['displacement'] = [0]
        x,_ = self.PS.x_v_current
        starting_positions = x.take(self.line_indices)

        converged = False
        while not converged:
            self.history['step'] += 1
            # Advance simulation
            self.PS.kin_damp_sim(forces)


            # Calculate and log displacement
            x,_ = self.PS.x_v_current
            displacement = x.take(self.line_indices) - starting_positions
            self.history['displacement'].append(np.mean(displacement))
            dx = np.mean(displacement)-self.history['displacement'][-2]
            # Check convergence
            if (self.history['step'] > self.params["min_iterations"]
                    and dx < self.params["convergence_threshold"]):
                converged = True
        finished_time = time.time()
        delta_time = finished_time - starting_time
        print(f'Finished at step {self.history["step"]} with {np.mean(displacement)=:.4f}')
        print(f'That took  {delta_time//60:.0f}m {delta_time%60:.2f}s')


class Simulate_airbag(Simulate):
    def __init__(self, ParticleSystem, params):
        # TODO set self.params to self.PS.params and modfiy all references to this previous behaviour
        self.PS = ParticleSystem
        self.params = params
        self.pressure = params['pressure']            # [Pa]


    def run_simulation(self,
                       plotframes: int = 0,
                       plot_whole_bag: bool = False,
                       printframes: int = 10,
                       simulation_function: str = 'default',
                       both_sides: bool = True
                       ):
        """


        Parameters
        ----------
        plotframes : INT, optional
            Save every nth frame. The default is 0, indicating saving zero frame.
        plot_whole_bag : bool, optional
            Wether or not to plot the whole bag. The default is False.
        printframes : int, optional
            Print a mesage every nth frame. The default is 10.
        simulation_function : str, optional
            Allows enabling kinetic damping by passing 'kinetic_damping'. The default is 'default'.
        both_sides : bool, optional
            Choose whether to mirror plot around x-y plane or not

        Returns
        -------
        None.

        """
        if simulation_function == 'kinetic_damping':
            simulation_function = self.PS.kin_damp_sim
        else:
            simulation_function = self.PS.simulate

        # Update pressure value for force calculation
        self.pressure = self.PS.params['pressure']            # [Pa]

        converged = False
        convergence_history = []
        dt = self.params['dt']

        if plotframes:
            fig = plt.figure()
        step = 0
        if hasattr(self.PS,'history'):
            step = len(self.PS.history['dt'])
        start_time = time.time()

        # setup convergence plot
        # fig_converge = plt.figure()
        # ax1 = fig_converge.add_subplot()
        # ax1.set_title('Convergence History')
        # plotline =  ax1.plot(convergence_history)[0]
        # ax1.set_yscale('log')

        while not converged:

            # Logic save plots of the simulation while it is running
            if plotframes and step%plotframes==0:
                # Live plot convergence history
                # plotline.set_ydata(convergence_history)
                # plotline.set_xdata(range(len(convergence_history)))
                # ax1.set_yscale('log')
                # fig_converge.canvas.draw()
                # fig_converge.canvas.flush_events()


                fig.clear()
                ax = fig.add_subplot(projection='3d')

                if plot_whole_bag:
                    self.plot_whole_airbag(ax, both_sides=both_sides)
                else:
                    self.PS.plot(ax)
                x,_ = self.PS.x_v_current_3D
                z = x[:,2]
                zlim =  np.max([z.max(), 1e-7])/100
                if plot_whole_bag:
                    ax.set_zlim(-zlim,zlim)
                else:
                    ax.set_zlim(0,zlim)
                t = np.sum(self.PS.history['dt'])
                ax.set_title(f"Simulate_airbag, t = {t:.5f}")
                fig.tight_layout()
                fig.savefig(f'temp\Airbag{step}.jpg', dpi = 200, format = 'jpg')

            # Force Calculation
            areas = self.PS.find_surface()
            areas = np.nan_to_num(areas)
            f = np.hstack(areas) * self.pressure

            # Advance 1 timesetp
            simulation_function(f)

            # Convergence checking
            d_crit_d_step = 0
            convergence_history.append(self.PS.kinetic_energy)
            if  len(convergence_history)>self.params['min_iterations']:
                d_crit_d_step = abs(convergence_history[-1]-convergence_history[-2])
                if d_crit_d_step<self.params['convergence_threshold']:
                    converged = True


            if printframes and step%printframes==0:
                current_time = time.time()
                t = current_time - start_time
                x,_ = self.PS.x_v_current_3D
                z_max = x[:,2].max()

                if 'dt' in self.PS.history:
                    dt = self.PS.history['dt'][-1]
                    print(f'{step=}, \tt={t//60:.0f}m {t%60:.2f}s, \tcrit={d_crit_d_step:.2g}, \t{z_max=:.4g}, \t{dt=:.2g}')
                else:
                    print(f'{step=}, \tt={t//60:.0f}m {t%60:.2f}s, \tcrit={d_crit_d_step:.2g}, \t{z_max=:.2g}')
            if step > self.params['t_steps']:
                converged = True
            step+= 1
        current_time = time.time()
        delta_time = current_time - start_time
        print(f'Converged in {delta_time//60:.0f}m {delta_time%60:.2f}s, {step} timesteps')

        convergence_history = np.array(convergence_history)
        self.PS.history['convergence'] = convergence_history
        if plotframes:
            fig_converge = plt.figure()
            ax1 = fig_converge.add_subplot()
            ax1.semilogy(convergence_history[convergence_history!=0])
            ax1.set_title('Convergence History')
        #print(convergence_history)

    def plot_whole_airbag(self,
                          ax = None,
                          plotting_function = 'default',
                          both_sides = True):
        """
        Plotting function that rotates and mirrors the simulated section

        Parameters
        ----------
        ax : matplotlib axis object
            Checks for preexisting axis. If none is given, one is made
        plotting_function : Callable, optional
            Allows for surface plot of the bag by passing 'surface'. The default is 'default'.
        both_sides : bool, optional
            Choose whether to mirror plot around x-y plane or not

        Returns
        -------
        ax : TYPE
            DESCRIPTION.

        """
        plotfunct_dict = {'default': self.PS.plot,
                          'surface': self.PS.plot_triangulated_surface}

        plotting_function = plotfunct_dict[plotting_function]
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        PS = self.PS

        x, _ = PS.x_v_current

        rotation_matrix = sp.spatial.transform.Rotation.from_euler('z', 90, degrees=True).as_matrix()
        rotation_matrix = sp.linalg.block_diag(*[rotation_matrix for i in range(int(len(x)/3))])

        for i in range(4):

            x = rotation_matrix.dot(x)
            PS.update_pos_unsafe(x)
            plotting_function(ax)
            if both_sides:
                x[2::3] *= -1
                PS.update_pos_unsafe(x)
                plotting_function(ax)

        return ax


class SimulateTripleChainWithMass(Simulate): # Simulate chain of  links joining in the  center where there is a large mass.
    def __init__(self, ParticleSystem, params):
        self.PS = ParticleSystem
        self.params = params


    def run_simulation(self):

        # Calculate external forces
        forces = -1*9.81*np.array([p.m for p in self.PS.particles])
        forces = np.outer(forces, [0,0,1]).flatten()

        converged = False
        self.convergence_history = {'e_kin': [],
                                    'x': [],
                                    'v': [],
                                    'f_int': []
                                    }
        self.PS.step = 0
        max_steps = self.params['max_sim_steps']
        info_dump_divisor = int(max_steps/100)
        while not converged:
            x,v = self.PS.kin_damp_sim(forces)

            self.PS.step+=1

            #internal_forces = [sd.force_value() for sd in self.PS.springdampers]
            #force_mag = np.linalg.norm(internal_forces, axis = 1)
            #residual = force_mag - np.mean(force_mag)
            #rms_res = np.sqrt(np.mean(residual*residual))

            #total_velocity = np.sum(np.array([p.v for p in self.PS.particles]))

            e_kin = self.PS.kinetic_energy
            f_int = [sd.force_value() for sd in self.PS.springdampers]

            self.convergence_history['e_kin'].append(e_kin)
            self.convergence_history['x'].append(x)
            self.convergence_history['v'].append(v)
            self.convergence_history['f_int'].append(f_int)

            if len(self.convergence_history['e_kin'])>100:
                recent_movement = np.mean(self.convergence_history['e_kin'][-10:-1])
                if self.PS.step%info_dump_divisor == 0:
                    logging.info(f'{self.PS.step=}, {recent_movement=}')
                if recent_movement<self.params['convergence_threshold']:
                    converged = True
            if self.PS.step>max_steps :
                logging.warning(f"Simulation exceeded limit with {self.PS.step=}>{max_steps=}")
                converged = True



class Simulate_Lightsail(Simulate):
    def __init__(self, ParticleSystem, ForceCalculator, params):
        # TODO set self.params to self.PS.params and modfiy all references to this previous behaviour
        self.PS = ParticleSystem
        self.params = params
        self.FC = ForceCalculator


    def run_simulation(self,
                       plotframes: int = 0,
                       printframes: int = 10,
                       simulation_function: str = 'default',
                       plot_forces=False,
                       file_id = ''):
        """
        Parameters
        ----------
        plotframes : INT, optional
            Save every nth frame. The default is 0, indicating saving zero frame.
        printframes : int, optional
            Print a mesage every nth frame. The default is 10.
        simulation_function : str, optional
            Allows enabling kinetic damping by passing 'kinetic_damping'. The default is 'default'.


        Returns
        -------
        None.

        """
        if simulation_function == 'kinetic_damping':
            simulation_function = self.PS.kin_damp_sim
        else:
            simulation_function = self.PS.simulate()

        converged = False
        convergence_history = []
        dt = self.params['dt']
        # We also want to log the forces, but logging all of them would be wastefull of memory
        buffer_size = 30
        if not 'forces_ringbuffer' in self.PS.history.keys():
            self.PS.history['forces_ringbuffer'] = [np.array([0]) for i in range(buffer_size)]

        if not 'net_force' in self.PS.history.keys():
            self.PS.history['net_force'] = []

        if plotframes:
            fig = plt.figure(figsize = [20,16])
        step = 0
        min_steps = self.params['min_iterations']
        if hasattr(self.PS,'history'):
            step = len(self.PS.history['dt'])
            min_steps += step

        start_time = time.time()

        # setup convergence plot
        # fig_converge = plt.figure()
        # ax1 = fig_converge.add_subplot()
        # ax1.set_title('Convergence History')
        # plotline =  ax1.plot(convergence_history)[0]
        # ax1.set_yscale('log')

        while not converged:
            # Force Calculation
            f = self.FC.force_value()
            net_force = np.sum(f, axis = 0)
            net_force = np.linalg.norm(net_force)
            self.PS.history['net_force'].append(net_force)

            # Logic save plots of the simulation while it is running
            if plotframes and step%plotframes==0:
                # Live plot convergence history
                # plotline.set_ydata(convergence_history)
                # plotline.set_xdata(range(len(convergence_history)))
                # ax1.set_yscale('log')
                # fig_converge.canvas.draw()
                # fig_converge.canvas.flush_events()


                fig.clear()
                ax = fig.add_subplot(projection='3d')
                if plot_forces:
                    self.PS.plot_forces(f,ax)
                else:
                    self.PS.plot(ax)
                x,_ = self.PS.x_v_current_3D
                z = x[:,2]
                zlim =  np.max([z.max(), 1e-7])
                ax.set_zlim(0,zlim)
                t = np.sum(self.PS.history['dt'])
                ax.set_title(f"Simulate Lightsail, t = {t:.5f}")
                fig.tight_layout()
                fig.savefig(f'temp\Lightsail{file_id}{step}.jpg', dpi = 200, format = 'jpg')

            # Advance 1 timestep
            simulation_function(f.ravel())

            # Convergence checking
            d_crit_d_step = 0
            #convergence_history.append(self.PS.kinetic_energy)
            convergence_history.append(net_force)
            self.PS.history['forces_ringbuffer'][step%buffer_size] = f
            if  step>min_steps:

                d_crit_d_step = abs(convergence_history[-1]-convergence_history[-2])
                if d_crit_d_step<self.params['convergence_threshold']:
                    converged = True


            if (printframes and step%printframes==0) or converged:
                current_time = time.time()
                t = current_time - start_time
                x,_ = self.PS.x_v_current_3D
                z_max = x[:,2].max()

                if 'dt' in self.PS.history:
                    dt = self.PS.history['dt'][-1]
                    print(f'{step=}, \tt={t//60:.0f}m {t%60:.2f}s, \tcrit={d_crit_d_step:.3g}, \t{z_max=:.4g}, \t{net_force=:.2g}, \t{dt=:.2g}')
                else:
                    print(f'{step=}, \tt={t//60:.0f}m {t%60:.2f}s, \tcrit={d_crit_d_step:.3g}, \t{z_max=:.2g}, \t{net_force=:.2g}')
            if step > self.params['t_steps']:
                converged = True
            step+= 1
        current_time = time.time()
        delta_time = current_time - start_time
        print(f'Converged in {delta_time//60:.0f}m {delta_time%60:.2f}s, {step} timesteps')

        convergence_history = np.array(convergence_history)
        if 'convergence' in self.PS.history.keys():
            self.PS.history['convergence'] = np.hstack((self.PS.history['convergence'],convergence_history))
        else:
            self.PS.history['convergence'] = convergence_history

        if plotframes:
            fig_converge = plt.figure()
            ax1 = fig_converge.add_subplot()
            ax1.semilogy(convergence_history[convergence_history!=0])
            ax1.set_title('Convergence History')
        #print(convergence_history)

    def simulate_trajectory(self,
                       plotframes: int = 0,
                       printframes: int = 10,
                       plot_angles: list = [-30,-60],
                       plot_forces= False,
                       plot_net_force = False,
                       file_id = '',
                       deform = True,
                       spin = True,
                       gravity = False,
                       damping = None,
                       initial_conditions = None):
        """
        Parameters
        ----------
        plotframes : INT, optional
            Save every nth frame. The default is 0, indicating saving zero frame.
        printframes : int, optional
            Print a mesage every nth frame. The default is 10.
        plot_angles : list
            length 2 list which holds elevation and azimuth
        plot_foces : bool
            Plot force vectors on the particle system?
        plot_net_force : bool
            Plot net force and moment over PS
        file_id : string
            allows to ad an ID infix in the filename to help distinguish plots from this sim run
        deform : bool
            allows to run a trajectory simulation without deformation
        spin : bool
            allows to run a trajectory simulation without rotation about z
        gravity : bool
            allows to enable gravitational force
        damping : npt.ArrayLike
            allows addition of damping forces to trajectory simulation. Should be length 6 array
            representing damping in [x,y,z,rx,ry,rz]
        initial_conditions : npt.Arraylike
            A 2x6 matrix representing the initial conditions for the simulation.
            The matrix should be structured as follows:
             - Row 1: Initial conditions for the variables [x, y, z, rx, ry, rz]
             - Row 2: Initial conditions for the first derivatives of the variables in Row 1
               [dx/dt, dy/dt, dz/dt, drx/dt, dry/dt, drz/dt]
        Returns
        -------
        None.

        """

        keys_1d = ['abs_force', "E_kin_xy", "E_kin_rot"]
        for key in keys_1d:
            if not key in self.PS.history.keys():
                self.PS.history[key] = np.zeros(int(self.params['t_steps']))

        keys_2d = ['net_force', 'net_moment', 'lin_accel', 'rot_accel']
        for key in keys_2d:
            if not key in self.PS.history.keys():
                self.PS.history[key] = np.zeros((int(self.params['t_steps']),3))
        keys_6d = ['position', 'velocity']
        for key in keys_6d:
            if not key in self.PS.history.keys():
                self.PS.history[key] = np.zeros((int(self.params['t_steps']),6))

        if plotframes:
            fig = plt.figure(figsize = [16,12])

        mass = sum([p.m for p in self.PS.particles])
        mmoi = np.sum(self.PS.calculate_mass_moment_of_inertia(),axis=0)
        f_grav = mass*-9.80665*np.array([0,0,1])
        attitude = np.zeros(3)
        length_scale = np.ptp(self.PS.x_v_current_3D[0][:,0])
        self.length_scale = length_scale
        f_max = np.max(self.FC.force_value())
        arrow_length = length_scale/f_max/4
        zlim = length_scale/5

        dt = self.params['dt']

        if type(initial_conditions) == type(None):
            v = np.zeros([6])
        else:
            self.PS.displace(initial_conditions[0], suppress_warnings = True)
            v = initial_conditions[1]
            v[2:] = np.deg2rad(v[2:])


        interpolators = set()
        if spin:
            for p in self.PS.particles:
                interpolators.add(p.optical_interpolator)

        step = 1
        min_steps = self.params['min_iterations']
        if hasattr(self.PS,'history'):
            step = len(self.PS.history['dt'])
            min_steps += step
        self.PS.history['position'][step,:3] = self.PS.calculate_center_of_mass()
        step +=1

        start_time = time.time()
        done = False
        while not done:
            if step > self.params['t_steps']-2:
                done = True
            # Force and Moment Calculation
            f = self.FC.force_value()
            net_force, net_moment  = self.FC.calculate_restoring_forces(forces=f)
            if type(damping)!=type(None):
                f_damping = v*damping
                net_force += f_damping[:3]
                net_moment += f_damping[3:]

            if gravity:
                lin_acceleration = (net_force+f_grav)/mass
            else:
                lin_acceleration = net_force/mass
            ang_acceleration= np.rad2deg(net_moment/mmoi)

            abs_force = np.linalg.norm(net_force)
            E_kin_xy = sum(1/2 * mass * v[:2]**2)
            E_kin_rot = sum(1/2 * mmoi * v[3:]**2)
            self.PS.history['abs_force'][step]=abs_force
            self.PS.history['net_force'][step]=net_force
            self.PS.history['net_moment'][step]=net_moment
            self.PS.history['lin_accel'][step]=lin_acceleration
            self.PS.history['rot_accel'][step]=ang_acceleration
            self.PS.history['E_kin_xy'][step]=E_kin_xy
            self.PS.history['E_kin_rot'][step]=E_kin_rot

            if 'dt' in self.PS.history:
                if len(self.PS.history['dt'])>0:
                    dt = self.PS.history['dt'][-1]
            COM = self.PS.calculate_center_of_mass()
            v += *(lin_acceleration*dt), *(ang_acceleration*dt)
            dx = v*dt
            if not spin:
                dx[-1]=0
            else: # update rotation of each interpolator to account for rotation of PhC
                for interp in interpolators:
                    interp.rotation+= dx[-1]
            attitude += dx[3:]

            self.PS.history['position'][step]=dx.copy()
            self.PS.history['velocity'][step]=v
            dx[2]=0 # Keep z at zero to keep it in frame

            self.PS.displace(dx, suppress_warnings=True)

            # Logic save plots of the simulation while it is running
            if plotframes and step%plotframes==0:
                fig.clear()
                ax = fig.add_subplot(projection='3d')
                if plot_forces:
                    self.PS.plot_forces(f,ax, length = arrow_length)
                else:
                    self.PS.plot(ax)
                ax.elev = plot_angles[0]
                ax.azim = plot_angles[1]
                if plot_net_force:
                    if not spin:
                        net_moment[2] =0
                    ax.quiver(COM[0],COM[1],COM[2],
                              net_force[0],net_force[1],net_force[2],
                              length = arrow_length/2e2, label ='Net Force', color='r')
                    ax.quiver(COM[0],COM[1],COM[2],
                              net_moment[0],net_moment[1],net_moment[2],
                              length = arrow_length*5e2, label ='Net Moment', color='magenta')
                    ax.legend()

                x,_ = self.PS.x_v_current_3D
                z = x[:,2]

                zlim = np.max([z.max(),zlim])
                ax.set_zlim(0,zlim)
                plot_scale=1
                ax.set_xlim([-length_scale*plot_scale,length_scale*plot_scale])
                ax.set_ylim([-length_scale*plot_scale,length_scale*plot_scale])
                t = np.sum(self.PS.history['dt'])
                ax.set_title(f"Simulate Lightsail, t = {t:.5f}")
                ax.set_aspect('equal')

                fig.tight_layout()
                fig.savefig(f'temp\Lightsail{file_id}{step}.jpg', dpi = 300, format = 'jpg')

            # Advance 1 timestep
            if deform:
                self.PS.simulate(f.ravel())
            else:
                self.PS.history['dt'].append(dt)


            if (printframes and step%printframes==0) or done:
                current_time = time.time()
                t = current_time - start_time
                location = np.round((COM/length_scale)[:2],4)
                angles = np.round(attitude,3)
                if 'dt' in self.PS.history:
                    print(f'{step=}, \tt={t//60:.0f}m {t%60:.2f}s, \t{abs_force=:.2g}, \t{dt=:.2g}, \t{location=} [D], \t{angles=} [deg], \t{E_kin_xy=:.2g}, \t{E_kin_rot=:.2g}'.replace('array',''))
                else:
                    print(f'{step=}, \tt={t//60:.0f}m {t%60:.2f}s, \t{abs_force=:.2g}, \t{location=} [D], \t{angles=} [deg]'.replace('array',''))
            # break if it flies off
            dx_recent = np.sum(np.abs(self.PS.history['position'][step-10:step][:,:2]))
            if step> min_steps and dx_recent<self.params['convergence_threshold']:
                done= True
                stable = True
            elif (abs(COM[0])>= length_scale or abs(COM[1])>= length_scale
                       or abs(attitude[0])>=10 or abs(attitude[1])>=10):
                COM/=length_scale
                print(f'Simulation halted: Lightsail broke perimiter {COM=} [D]')
                done = True
                stable = False


            step+= 1

        current_time = time.time()
        delta_time = current_time - start_time
        print(f'Converged in {delta_time//60:.0f}m {delta_time%60:.2f}s, {step} timesteps')
        return stable

        if plotframes:
            self.plot_flight_hist()

    def plot_flight_hist(self, energy = False, pos_offset = [0,0]):
        length_scale = self.length_scale
        time_history = np.cumsum(self.PS.history['dt'], axis=0)
        position_history = np.cumsum(self.PS.history['position'][:len(time_history)], axis=0)
        velocity_history = np.array(self.PS.history['velocity'][:len(time_history)])
        r = np.sqrt(position_history[:,0]**2 + position_history[:,1]**2)

        position_history[:,0] -= pos_offset[0]
        position_history[:,1] -= pos_offset[1]

        E_kin_xy = self.PS.history['E_kin_xy'][:len(time_history)]
        E_kin_rot = self.PS.history['E_kin_rot'][:len(time_history)]

        fig_movement = plt.figure(figsize=[10,8])
        ax1 = fig_movement.add_subplot(231)
        ax1.plot(position_history[:,0]/length_scale,position_history[:,1]/length_scale,
                 marker='*', lw=0.5, ms=2)
        ax1.set_title('X-Y trajectory')
        ax1.set_xlabel('X [D]')
        ax1.set_ylabel('Y [D]')
        #ax1.set_xlim([-3,3])
        #ax1.set_ylim([-3,3])
        #ax1.set_aspect('equal')
        ax1.grid()

        ax2 = fig_movement.add_subplot(232)
        ax2.plot(position_history[:,0]/length_scale,position_history[:,4],
                 marker='*', lw=0.5, ms=2)
        ax2.set_title('X-$\\theta_{y}$ trajectory')
        ax2.set_xlabel('X [D]')
        ax2.set_ylabel('$\\theta_{y}$ [deg]')
        ax2.grid()

        ax3 = fig_movement.add_subplot(236)
        ax3.plot(r/length_scale, position_history[:,2]/length_scale,
                 marker='*', lw=0.5, ms=2)
        ax3.set_title('r-z trajectory')
        ax3.set_xlabel('r [D]')
        ax3.set_ylabel('z [D]')
        ax3.grid()

        ax4 = fig_movement.add_subplot(234)
        ax4.plot(time_history,position_history[:,0]/length_scale, label = 'X',
                 marker='*', lw=0.5, ms=2)
        ax4.plot(time_history,position_history[:,1]/length_scale, label = 'Y',
                 marker='*', lw=0.5, ms=2)
        ax4.set_title('x and y location over time')
        ax4.set_xlabel('t [t]')
        ax4.set_ylabel('distance [D]')
        ax4.legend()
        ax4.grid()

        ax5 = fig_movement.add_subplot(235)
        ax5.plot(position_history[:,3], position_history[:,4],
                 marker='*', lw=0.5, ms=2)
        ax5.set_ylabel('$\\theta_{y}$ [deg]')
        ax5.set_xlabel('$\\theta_{x}$ [deg]')
        ax5.grid()

        ax6 = fig_movement.add_subplot(233)
        ax6.plot(position_history[:,1]/length_scale,position_history[:,3],
                 marker='*', lw=0.5, ms=2)
        ax6.set_title('Y-$\\theta_{x}$ trajectory')
        ax6.set_xlabel('Y [D]')
        ax6.set_ylabel('$\\theta_{x}$ [deg]')
        ax6.grid()

        fig_movement.tight_layout()

        if energy:
            fig_velocity = plt.figure()

            ax7 = fig_velocity.add_subplot()
            ax7.set_title('Kinetic Energy')
            ax7.plot(time_history, E_kin_xy, label ='lin',
                     marker='*', lw=0.5, ms=2)
            ax7.plot(time_history, E_kin_rot, label='rot',
                     marker='*', lw=0.5, ms=2)
            ax7.set_xlabel('Time [s]')
            ax7.set_ylabel('Energy [J]')
            ax7.legend()
            ax7.grid()


if __name__ == '__main__':
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
        "max_iter": 1e2,  # [-]       maximum number of iterations]

        # Simulation Steps
        "steps": np.linspace(0.01,0.1, 25),

        # Mesh_dependent_settings
        "midstrip_width": 1,
        "boundary_margin": 0.175
        }

    mesh = MF.mesh_square_cross(30,30,1,params)
    # initial_conditions, connections = MF.mesh_rotate_and_trim(initial_conditions,
    #                                                        connections,
    #                                                        45/2)
    PS = ParticleSystem(*mesh,params)

    Sim = Simulate_1d_Stretch(PS, params)
    Sim.run_simulation()
    Sim.plot_results()
    for key in Sim.history.keys():
        reaction, poisson = Sim.history[key]
        print(f"Strain {key}, {reaction=}, {poisson=}")

