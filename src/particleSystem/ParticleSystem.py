"""
ParticleSystem framework
...
"""
import logging 

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import bicgstab
import scipy.sparse as sps
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from src.particleSystem.Particle import Particle
from src.particleSystem.SpringDamper import SpringDamper 


class ParticleSystem:
    def __init__(self, 
                 connectivity_matrix: list, 
                 initial_conditions: npt.ArrayLike, 
                 sim_param: dict,
                 clean_particles: bool = True):
        """
        Constructor for ParticleSystem object, model made up of n particles
        
        Parameters
        ---------
        connectivity_matrix : list
            2-by-m matrix, where each column contains a nodal index pair that 
            is connectedby a spring element:
                [ p1: Particle, p2: Particle, k: float, c: float, optional : linktype]
        initial_conditions :  npt.ArrayLike
            Array of n arrays to instantiate particles. Each subarray must 
            contain the params required for the particle constructor: 
                [initial_pos, initial_vel, mass, fixed: bool, constraint, optional : constraint_type]
        param sim_param : dict
            Dictionary of other parameters required for simulation (dt, rtol, ...)
        clean_particles : bool
            Sets wether or not to delete particles without connections on init
        """
        if clean_particles:
            self.clean_up(connectivity_matrix, initial_conditions)
        
        self.__connectivity_matrix = connectivity_matrix

        self.__n = len(initial_conditions)
        self.__dt = sim_param["dt"]
        self.__rtol = sim_param["rel_tol"]
        self.__atol = sim_param["abs_tol"]
        self.__maxiter = sim_param["max_iter"]

        # allocate memory
        self.__particles = []
        self.__springdampers = []
        self.__f = np.zeros((self.__n * 3, ),dtype='float64')
        self.__jx = np.zeros((self.__n * 3, self.__n * 3))
        self.__jv = np.zeros((self.__n * 3, self.__n * 3))

        self.__instantiate_particles(initial_conditions)
        self.__m_matrix = self.__construct_m_matrix()
        self.__instantiate_springdampers()

        # Variables required for kinetic damping
        self.__w_kin = self.__calc_kin_energy()
        self.__w_kin_min1 = self.__calc_kin_energy()
        self.__vis_damp = True
        self.__x_min1 = np.zeros(self.__n, )
        self.__x_min2 = np.zeros(self.__n, )
        return

    def __str__(self):
        description = ""
        description +="ParticleSystem object instantiated with attributes\nConnectivity matrix:"
        description += str(self.__connectivity_matrix)
        description +="\n\nInstantiated particles:\n"
        n = 1
        for particle in self.__particles:
            description += f"p{n}: {particle}\n"
            n += 1
        return description

    def __instantiate_particles(self, initial_conditions: list):
        for set_of_initial_cond in initial_conditions:
            x = set_of_initial_cond[0]
            v = set_of_initial_cond[1]
            m = set_of_initial_cond[2]
            f = set_of_initial_cond[3]
            if f and len(set_of_initial_cond)>=5:
                con = set_of_initial_cond[4]
                con_t = set_of_initial_cond[5]
                self.__particles.append(Particle(x, v, m, f, con, con_t))
            else:
                self.__particles.append(Particle(x, v, m, f))
        return

    def __instantiate_springdampers(self):
        for link in self.__connectivity_matrix:
            link = link.copy() #needed to not override the __connectivity_matrix
            link[0] = self.__particles[link[0]]
            link[1] = self.__particles[link[1]]
            SD = SpringDamper(*link)
            self.__springdampers.append(SD)
            link[0].connections.append(SD)
            link[1].connections.append(SD)
        return
    
    def clean_up(self, connectivity_matrix, initial_conditions):
        remove_list = set(range(len(initial_conditions)))
        for link in connectivity_matrix:
            try: 
                remove_list.remove(link[0])
            except:
                pass
            try: 
                remove_list.remove(link[1])
            except:
                pass
        remove_list = list(remove_list)
        remove_list.sort()
        for i in remove_list[::-1]:
            del initial_conditions[i]
            for link in connectivity_matrix:
                if link[0]>i:
                    link[0]-=1
                if link[1]>i:
                    link[1]-=1
    
    def stress_self(self, factor: float = 0):
        """Set all node lengths to zero to homogenously stress mesh"""
        if factor == 0:
            for link in self.springdampers:
                link.l0 = 0
        else:
            for link in self.springdampers:
                link.l0 *= factor 
                
        return
        

    def __construct_m_matrix(self):
        matrix = np.zeros((self.__n * 3, self.__n * 3))

        for i in range(self.__n):
            matrix[i*3:i*3+3, i*3:i*3+3] += np.identity(3)*self.__particles[i].m

        return matrix

    def __calc_kin_energy(self):
        v = self.__pack_v_current()
        w_kin = np.matmul(np.matmul(v, self.__m_matrix), v.T)      # Kinetic energy, 0.5 constant can be neglected
        return w_kin

    def simulate(self, f_external: npt.ArrayLike = ()):
        if not len(f_external):             # check if external force is passed as argument, otherwise use 0 vector
            f_external = np.zeros(self.__n * 3, )
        f = self.__one_d_force_vector() + f_external

        v_current = self.__pack_v_current()
        x_current = self.__pack_x_current()

        jx, jv = self.__system_jacobians()
        
        #jx = sps.lil_array(jx)
        #jv = sps.lil_array(jv)
        
        # constructing A matrix and b vector for solver
        A = self.__m_matrix - self.__dt * jv - self.__dt ** 2 * jx
        b = self.__dt * f + self.__dt ** 2 * jx.dot(v_current)

        # checking conditioning of A
        # print("conditioning A:", np.linalg.cond(A))
        #A = sps.bsr_array(A)
        # BiCGSTAB from scipy library
        dv, _ = bicgstab(A, b, tol=self.__rtol, atol=self.__atol, maxiter=self.__maxiter)

        # numerical time integration following implicit Euler scheme
        v_next = v_current + dv
        x_next = x_current + self.__dt * v_next

        # function returns the pos. and vel. for the next timestep, but for fixed particles this value doesn't update!
        self.__update_x_v(x_next, v_next)
        return x_next, v_next

    def kin_damp_sim(self, f_ext: npt.ArrayLike = (), q_correction: bool = False):       # kinetic damping algorithm
        if self.__vis_damp:         # Condition resetting viscous damping to 0
            for link in self.__springdampers:
                link.c = 0
            self.__c = 0
            self.__vis_damp = False

        if len(f_ext):              # condition checking if an f_ext is passed as argument
            self.__save_state()
            x_next, v_next = self.simulate(f_ext)
        else:
            self.__save_state()
            x_next, v_next = self.simulate()

        w_kin_new = self.__calc_kin_energy()

        if w_kin_new > self.__w_kin:    # kin damping algorithm, takes effect when decrease in kin energy is detected
            self.__update_w_kin(w_kin_new)
        else:
            v_next = np.zeros(self.__n*3, )

            if q_correction:            # statement to check if q_correction is desired, standard is turned off
                q = (self.__w_kin - w_kin_new)/(2*self.__w_kin - self.__w_kin_min1 - w_kin_new)
                # print(q)
                # print(self.__w_kin, w_kin_new)
                # !!! Not sure if linear interpolation between states is the way to determine new x_next !!!
                if q < 0.5:
                    x_next = self.__x_min2 + (q / 0.5) * (self.__x_min1 - self.__x_min2)
                elif q == 0.5:
                    x_next = self.__x_min1
                elif q < 1:
                    x_next = self.__x_min1 + ((q - 0.5) / 0.5) * (x_next - self.__x_min1)

                # Can also use this q factor to recalculate the state for certain timestep h

            self.__update_x_v(x_next, v_next)
            self.__update_w_kin(0)

        return x_next, v_next

    def __pack_v_current(self):
        return np.array([particle.v for particle in self.__particles]).flatten()

    def __pack_x_current(self):
        return np.array([particle.x for particle in self.__particles]).flatten()

    def __one_d_force_vector(self):
        self.__f[self.__f != 0] = 0

        for n in range(len(self.__springdampers)):
            f_int = self.__springdampers[n].force_value()
            i, j, *_ = self.__connectivity_matrix[n]

            self.__f[i*3: i*3 + 3] += f_int
            self.__f[j*3: j*3 + 3] -= f_int

        return self.__f

    def __system_jacobians(self):
        self.__jx[self.__jx != 0] = 0
        self.__jv[self.__jv != 0] = 0

        for n in range(len(self.__springdampers)):
            jx, jv = self.__springdampers[n].calculate_jacobian()
            i, j, *_ = self.__connectivity_matrix[n]
            if self.__particles[i].fixed:
                if self.__particles[i].constraint_type == 'point': 
                    jxplus = np.zeros([3,3])
                    jvplus = jxplus
                else:
                    jxplus = self.__particles[i].constraint_projection_matrix.dot(jx)
                    jvplus = self.__particles[i].constraint_projection_matrix.dot(jv)
            else: 
                jxplus = jx
                jvplus = jv
            
            if self.__particles[j].fixed:
                if self.__particles[j].constraint_type == 'point': 
                    jxmin = np.zeros([3,3])
                    jvmin = jxmin
                else:
                    jxmin = self.__particles[j].constraint_projection_matrix.dot(jx)
                    jvmin = self.__particles[j].constraint_projection_matrix.dot(jv)
            else: 
                jxmin = jx
                jvmin = jv
            
            self.__jx[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jxplus
            self.__jx[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jxplus
            self.__jx[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jxmin
            self.__jx[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jxmin

            self.__jv[i * 3:i * 3 + 3, i * 3:i * 3 + 3] += jvplus
            self.__jv[j * 3:j * 3 + 3, j * 3:j * 3 + 3] += jvplus
            self.__jv[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= jvmin
            self.__jv[j * 3:j * 3 + 3, i * 3:i * 3 + 3] -= jvmin

        return self.__jx, self.__jv

    def __update_x_v(self, x_next: npt.ArrayLike, v_next: npt.ArrayLike):
        for i in range(self.__n):
            self.__particles[i].update_pos(x_next[i * 3:i * 3 + 3])
            self.__particles[i].update_vel(v_next[i * 3:i * 3 + 3])
        return

    def __update_w_kin(self, w_kin_new: float):
        self.__w_kin_min1 = self.__w_kin
        self.__w_kin = w_kin_new
        return
    
    def update_pos_unsafe(self, x_new: npt.ArrayLike):
        for i, particle in enumerate(self.__particles):
            particle.update_pos_unsafe(x_new[3*i: 3*i+3])

    def __save_state(self):
        self.__x_min2 = self.__x_min1
        self.__x_min1 = self.__pack_x_current()
        return
    
    def find_reaction_forces(self):
        fixlist = [p.fixed for p in self.particles]
        projections = [p.constraint_projection_matrix for p in np.array(self.particles)[fixlist]]
        forces = self.__f.reshape((self.__n,3))
        forces = -forces[fixlist]
        for i, projection in enumerate(projections):
            forces[i] -= projection.dot(forces[i].T).T
        return forces

    @property
    def particles(self):            # @property decorators required, as PS info might be required for external calcs
        return self.__particles

    @property
    def springdampers(self):
        return self.__springdampers

    # @property
    # def stiffness_m(self):
    #     self.__system_jacobians()
    #     return self.__jx
    
    @property
    def kinetic_energy(self):
        return self.__calc_kin_energy()

    @property
    def f_int(self):
        f_int = self.__f.copy()
        for i in range(len(self.__particles)):      # need to exclude fixed particles for force-based convergence
            if self.__particles[i].fixed:
                f_int[i*3:(i+1)*3] = 0

        return f_int

    @property
    def x_v_current(self):
        return self.__pack_x_current(), self.__pack_v_current()
    
    @property
    def x_v_current_3D(self):
        x = self.__pack_x_current()
        v = self.__pack_v_current()
        x = np.reshape(x, (int(len(x)/3),3))
        v = np.reshape(v, (int(len(v)/3),3))
        return x, v

    def plot(self, ax=None):
        """"Plots current system configuration"""
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        
        
        fixlist = []
        freelist = []
        for particle in self.__particles:
            if particle.fixed:
                fixlist.append(particle.x)
            else:
                freelist.append(particle.x)
                
        fixlist = np.array(fixlist)
        freelist = np.array(freelist)
        
        if len(fixlist)>0:
            ax.scatter(fixlist[:,0],fixlist[:,1],fixlist[:,2], color = 'red', marker = 'o')
        if len(freelist)>0:
            ax.scatter(freelist[:,0],freelist[:,1],freelist[:,2], color = 'blue', marker = 'o', s =5)
        
        segments = []
        
        for link in self.__springdampers:
            segments.append(link.line_segment())
            
        lc = Line3DCollection(segments, colors = 'black', linewidths = 0.5)  
        ax.add_collection3d(lc)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect([1,1,1])
        
        return ax
    
    def initialize_find_surface(self, projection_plane: str = 'z'):
        """
        performs triangulation and sets up conversion matrix for surface calc
        
        Projects the point cloud onto specified plane and performs 
        triangulation. Then uses shape of current triangles to create a 
        conversion matrix for assigning the areas of each triangle onto the 
        nodes. 
        
        Parameters
        ----------
        projection_plane : str
            normal direction of plane for the mesh to be projected on for 
            triangulation. Default: z
        
        
        Returns
        -------
        simplices : list
            nested list of node indices that make up triangles
        conversion_matrix : npt.ArrayLike
            ndarray of shape n_nodes x n_triangles

        """
        # Gathering points of nodes
        points = self.__pack_x_current()
        points = points.reshape((int(len(points)/3),3))
        
        # Checking projection plane
        if projection_plane == 'x':
            projection_plane = 0
        elif projection_plane == 'y':
            projection_plane = 1
        elif projection_plane == 'z':
            projection_plane = 2
        else:
            raise AttributeError("projection_plane improperly defined; Must be x, y or z.")
        
        # Performing triangulation
        points_projected = points[:,:projection_plane] # Projecting onto x-y plane
        tri = Delaunay(points_projected)
        
        # Finding areas of each triangle
        v1 = points[tri.simplices[:,0]]-points[tri.simplices[:,1]]
        v2 = points[tri.simplices[:,0]]-points[tri.simplices[:,2]]
        
        # Next we set up the matrix multiplication that will divide the areas 
        # of the triangles over the actual nodes
        #conversion_matrix = np.zeros((self.__n*3,len(tri.simplices)*3))

        v1_length = np.linalg.norm(v1, axis=1)
        v2_length = np.linalg.norm(v2, axis=1)
        v3_length = np.linalg.norm(v2-v1, axis=1)

        angle_1 = np.arccos(np.sum(v1*v2, axis = 1)/(v1_length*v2_length))
        
        # Next bit is a fix for an error due to limited numerical accuracy
        inp = v2_length/v3_length * np.sin(angle_1)
        inp[inp>1] = 1
        angle_2 = np.arcsin(inp)
        angle_3 = np.pi - angle_1 - angle_2

        angle_iterator = np.column_stack((angle_1, angle_2, angle_3)).flatten()/np.pi
        
        # Sparse matrix construction
        rows = []
        cols = []
        data = []
        for j, indices in enumerate(tri.simplices):
            for k, i in enumerate(indices):
                for l in range(3):
                    rows.append(3*i+l)
                    cols.append(3*j+l)
                    data.append(angle_iterator[3*j+k])
        conversion_matrix = sps.csr_matrix((data, (rows, cols)), shape=(self.__n*3, len(tri.simplices)*3))
        
        #for j, indices in enumerate(tri.simplices):
        #    for k, i in enumerate(indices):
        #        conversion_matrix[3*i,3*j]+= angle_iterator[3*j+k]
        #        conversion_matrix[3*i+1,3*j+1]+= angle_iterator[3*j+k]
        #        conversion_matrix[3*i+2,3*j+2]+= angle_iterator[3*j+k]
        
        
        self.__simplices = tri.simplices 
        self.__surface_conversion_matrix = conversion_matrix
        
        return tri.simplices, conversion_matrix
    
    def find_surface(self, projection_plane: str = 'z') -> np.ndarray:
        """
        finds the surface area vector for each node in the mesh
        
        Parameters
        ----------
            projection_plane: passed to self.initialize_find_surface().
            
        
        Returns
        -------
        areas: npt.ArrayLike 
            3D area vectors for each node
        
        """
        
        if not hasattr(self, '_ParticleSystem__surface_conversion_matrix'):
            logging.warning('find_surface called without prior initialization.')
            simplices, conversion_matrix = self.initialize_find_surface(projection_plane)
            self.__simplices = simplices 
            self.__surface_conversion_matrix = conversion_matrix
        else: 
            conversion_matrix = self.__surface_conversion_matrix
            simplices = self.__simplices
        
        # Gathering points of nodes
        points = self.__pack_x_current()
        n = len(points)
        points = points.reshape((int(n/3),3))
        
        # Finding areas of each triangle
        v1 = points[simplices[:,0]]-points[simplices[:,1]]
        v2 = points[simplices[:,0]]-points[simplices[:,2]]
        
        # Calculate the area of the triangulated simplices
        area_vectors = np.cross(v1,v2)/2
        
        # Convert these to correct particle area magnitudes
        # Summing vectors oposing directions cancel, which we need for finding
        # the direction but diminishes the area magnitude. We need to correct
        # for this by calculating them seperately and scaling the vector.
        simplice_area_magnitudes = np.linalg.norm(area_vectors, axis=1)
        logging.debug(f'{np.sum(simplice_area_magnitudes)=}')
        
        simplice_area_magnitudes_1d = np.outer(simplice_area_magnitudes,np.ones(3)).flatten()
        particle_area_magnitudes_1d = conversion_matrix.dot(simplice_area_magnitudes_1d)
        logging.debug(f'{np.sum(particle_area_magnitudes_1d)=}')
        logging.debug(f'{np.sum(particle_area_magnitudes_1d[::3])=}')
        
        # Now we transorm the simplice areas into nodal areas
        input_vector = area_vectors.flatten()
        area_vectors_1d_direction = conversion_matrix.dot(input_vector)
        area_vectors_redistributed = area_vectors_1d_direction.reshape((int(n/3),3))
        
        # Scaling the vectors        
        direction_magnitudes = np.linalg.norm(area_vectors_redistributed, axis = 1)
        logging.debug(f'{np.sum(direction_magnitudes)=}')
        
        scaling_factor = particle_area_magnitudes_1d[::3] /direction_magnitudes
        logging.debug(f'{scaling_factor=}')
        
        area_vectors_redistributed *= np.outer(scaling_factor,np.ones(3))
        
        logging.debug(f'After scaling {np.sum(np.linalg.norm(area_vectors_redistributed, axis=1))=}')

        
        return area_vectors_redistributed
    
    def plot_triangulated_surface(self, ax = None):
        """
        plots triangulated surface for user inspection

        """
        
        # Gathering points of nodes
        points = self.__pack_x_current()
        points = points.reshape((int(len(points)/3),3))
        x,y,z = points[:,0], points[:,1], points[:,2]

        
        area_vectors = self.find_surface()
        a_u = area_vectors[:,0]
        a_v = area_vectors[:,1]
        a_w = area_vectors[:,2]
        
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(x, y, z, triangles=self.__simplices, cmap=plt.cm.Spectral)
        ax.scatter(x,y,z)
        ax.quiver(x,y,z,a_u,a_v,a_w, length = 1)

        return ax
    
    def calculate_correct_masses(self, thickness, density):
        areas = np.linalg.norm(self.find_surface(), axis=1)
        masses = areas * thickness * density
        for i, particle in enumerate(self.particles):
            particle.set_m(masses[i])
        
        # Recalculate mass matrix
        self.__m_matrix = self.__construct_m_matrix()

        


if __name__ == "__main__":
    params = {
        # model parameters
        "n": 3,  # [-] number of particles
        "k": 2e4,  # [N/m] spring stiffness
        "c": 0,  # [N s/m] damping coefficient
        "l0": 0,  # [m] rest length

        # simulation settings
        "dt": 0.001,  # [s] simulation timestep
        "t_steps": 1000,  # [-] number of simulated time steps
        "abs_tol": 1e-50,  # [m/s] absolute error tolerance iterative solver
        "rel_tol": 1e-5,  # [-] relative error tolerance iterative solver
        "max_iter": 1e5,  # [-] maximum number of iterations

        # physical parameters
        "g": 9.81           # [m/s^2] gravitational acceleration
    }
    c_matrix = [[0, 1, params['k'], params['c']],
                [1, 2, params['k'], params['c']]
                ]
    init_cond = [[[0, 0, 0], [0, 0, 0], 1, True],
                 [[1, 0, 0], [0, 0, 0], 1, False],
                 [[1, 1, 0], [0, 0, 0], 1, False]
                 ]


    ps = ParticleSystem(c_matrix, init_cond, params)
    print(ps)
    ps.plot()
    ps.plot_triangulated_surface()
