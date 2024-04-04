# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:42:01 2024

@author: Mark Kalsbeek
"""
import time
dump =[]
t_last = time.time()
for i in [*range(3,40),*range(3,40),*range(3,40)]:
    Sim = setup_testcase(n_segments = i, params = params_square_high_t)
    Sim.run_simulation(plotframes=0, 
                       plot_whole_bag = False,
                       printframes=0,
                       simulation_function = 'kinetic_damping',
                       both_sides=False)
    PS = Sim.PS
    x, v = PS.x_v_current_3D
    z_max = x[:,2].max()
    
    t_current = time.time()
    dt = t_current-t_last
    t_last = t_current
    dump.append([i,z_max, dt])
    print(dump[-1])
    
    
dump= np.array(dump)
plt.plot(dump)
plt.xlabel('Number of segments')
plt.ylabel('Peak displacement')
plt.title('Mesh refinement versus displacement')
plt.margins(0,0.1)
