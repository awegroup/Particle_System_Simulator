import numpy as np
from pyfe3d import Spring, SpringProbe

class SpringElement:
    def __init__(self, n1 : int, n2 : int, init_k_KC0 : int):
        self.DOF = 6
        springprobe = SpringProbe()
        self.spring = Spring(springprobe)
        self.spring.init_k_KC0 = init_k_KC0
        self.spring.n1 = n1
        self.spring.n2 = n2
        self.spring.c1 = self.DOF * n1
        self.spring.c2 = self.DOF * n2
        self.update_KC0v_only = 0
        
    def set_spring_properties(self, l0 : float, k : float, springtype : str):
        self.l0 = l0 
        self.k = k
        self.spring.kxe = k  
        self.springtype = springtype.lower()
        if self.springtype not in ("noncompressive", "default", "pulley"):
            raise ValueError("Invalid spring type. Choose from 'noncompressive', 'default', or 'pulley'.")
        
    def unit_vector(self, ncoords : np.ndarray):
        xi = ncoords[self.spring.c2//2 + 0] - ncoords[self.spring.c1//2 + 0]
        xj = ncoords[self.spring.c2//2 + 1] - ncoords[self.spring.c1//2 + 1]
        xk = ncoords[self.spring.c2//2 + 2] - ncoords[self.spring.c1//2 + 2]
        l = (xi**2 + xj**2 + xk**2)**0.5
        unit_vect = np.array([xi, xj, xk])/l
        return unit_vect,l
    
    def update_KC0(self, KC0r : np.ndarray, KC0c : np.ndarray, KC0v : np.ndarray, ncoords : np.ndarray):
        unit_vect,l = self.unit_vector(ncoords)
        xi, xj ,xk = unit_vect[0], unit_vect[1], unit_vect[2]      
        vxyi, vxyj, vxyk =  unit_vect[1], unit_vect[2], unit_vect[0] 
        if xi == xj  and xj == xk: # Edge case, if all are the same then KC0 returns NaN's
            vxyi *= -1
        self.spring.update_rotation_matrix(xi, xj, xk, vxyi, vxyj, vxyk)
        self.spring.update_KC0(KC0r, KC0c, KC0v, self.update_KC0v_only)
        self.update_KC0v_only = 1
        return KC0r, KC0c, KC0v
    
    def spring_internal_forces(self, ncoords: np.ndarray, l_other_pulley: float = 0):
        unit_vector,l = self.unit_vector(ncoords)
        if self.springtype == "noncompressive" or self.springtype == "pulley":
            if (l + l_other_pulley) < self.l0:
                self.spring.kxe = 0
            else:
                self.spring.kxe = self.k
        pulley_factor = l / (l + l_other_pulley)
        f_s = self.spring.kxe  * (l - pulley_factor * self.l0) # TODO fi =x k for noncompressive pulley
        fi = f_s * unit_vector
        fi = np.append(fi, [0, 0, 0])

        return fi
    

if __name__ == "__main__":
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import coo_matrix
    from pyfe3d import DOF, INT, DOUBLE, SpringData
    springdata = SpringData()
    init_k_KC0 = 0
    nids = np.array([0, 1])  # Node IDs
    num_elements = len(nids)-1
    KC0r = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=INT)  # Two springs
    KC0c = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=INT)
    KC0v = np.zeros(springdata.KC0_SPARSE_SIZE * num_elements, dtype=DOUBLE)
    N = DOF * len(nids)  # Total DOFs
    n1 = 0
    n2 = 1
    ncoords = np.array([[0, 0, 0],[1, 0, 0]],dtype=DOUBLE) 
    ncoords_init = ncoords.flatten()  # Flatten the coordinates for the spring element
    SpringElement1 = SpringElement(n1, n2, init_k_KC0)
    SpringElement1.set_spring_properties(l0=0, k=1, springtype="Default")
    SpringElement1.update_KC0(KC0r, KC0c, KC0v, ncoords_init)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    bk = np.ones(N, dtype=bool)
    bk[DOF] = False 
    bu = ~bk
    f = np.zeros(N)
    f[DOF] = 2  
    KC0uu = KC0[bu, :][:, bu]
    fu = f[bu]  # Free DOFs force vector
    fi = np.zeros(N, dtype=DOUBLE)
    fi_global = SpringElement1.spring_internal_forces(ncoords_init)
    bu1 = bu[SpringElement1.spring.c1:SpringElement1.spring.c1+DOF]
    bu2 = bu[SpringElement1.spring.c2:SpringElement1.spring.c2+DOF]
    fi[SpringElement1.spring.n1*DOF:(SpringElement1.spring.n1+1)*DOF] -= fi_global*bu1
    fi[SpringElement1.spring.n2*DOF:(SpringElement1.spring.n2+1)*DOF] += fi_global*bu2
    
    residual = fu - fi[bu]  # Calculate the residual
    print("residual", residual)
    uu = spsolve(KC0uu, residual)  # Solve for displacements
    u = np.zeros(N)
    u[bu] = uu  # Fill in the displacements for free DOFs
    xyz = np.zeros(N, dtype=bool)  # Initialize xyz array
    xyz[0::DOF] = xyz[1::DOF] = xyz[2::DOF] = True  
    ncoords_current = ncoords_init.copy()
    ncoords_current += u[xyz]  # Update node coordinates with dis
    print(ncoords_current)
