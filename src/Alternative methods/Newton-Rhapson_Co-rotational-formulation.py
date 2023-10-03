# Temp used for graph generation
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt

# " Figure that illustrates the thought behind kinetic damping"
# t = np.linspace(0, 10, 1000)
#
# pos = 2*np.cos(t)
# ke = np.cos(t)
# pe = np.sin(t)
#
# plt.plot(t, pos)
# plt.plot(t, np.abs(ke))
# plt.plot(t, np.abs(pe))
# plt.axvline(x=np.pi/2, ymin=0.5, ymax=0.72,  ls='--')
# plt.axvline(x=np.pi*2.5, ymin=0.5, ymax=0.72,  ls='--')
# plt.axvline(x=np.pi*1.5, ymin=0.5, ymax=0.72,  ls='--')
#
# plt.grid()
# plt.title('normalized energies for position of simple harmonic oscillator')
# plt.xlabel('time [s]')
# plt.ylabel('position [m]')
# plt.legend(['position', 'potential energy', 'kinetic energy'])
#
# plt.show()

"Figure to illustrate the computation of the quadratic correction value"

t = np.linspace(0, np.pi-0.5, 1000)
ke = np.sin(t)

fig, ax = plt.subplots(1)
plt.plot(t, np.abs(ke))

# plt.axvline(x=np.pi/2, ymin=0, ymax=0.95,  ls='--', color='red')

plt.grid()
plt.title('')
plt.xlabel('time')
plt.ylabel('kinetic energy')
# plt.legend(['assumed KE quadratic', 'KE peak', 'kinetic energy'])

# ax.set_yticklabels([np.pi/2], labels=["t*"])
xticks = list(np.linspace(0, np.pi-0.5, 3))
xticks.append(np.pi/2)
xlabels = ['t-2h', 't-h', 't', r'$t^*$']

yticks = []
yticks.append(np.sin(np.pi/2))
ylabels = [r'$W^*$']

ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_xticklabels(xlabels)

plt.show()