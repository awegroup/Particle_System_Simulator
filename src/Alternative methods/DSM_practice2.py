import math
import copy
import numpy as np
import matplotlib.pyplot as plt


E = 1#200e9
A = 1#0.005

nodes = [[0, 0, 0, 0], [1, 0, 0, 5], [2, 0, 0, 10]]
elements = [[0, 0, 1], [1, 1, 2]]

theta = 0
theta2 = 2.2143
L = 3
L2 = 5

def s_m (theta, E, A, L):
    k = E * A / L
    e11 = math.cos(theta) ** 2
    e12 = math.cos(theta) * math.sin(theta)
    e21 = math.cos(theta) * math.sin(theta)
    e22 = math.sin(theta) ** 2

    K11 = k * np.array([[e11, e12], [e21, e22]])
    K12 = k * np.array([[-e11, -e12], [-e21, -e22]])
    K21 = k * np.array([[-e11, -e12], [-e21, -e22]])
    K22 = k * np.array([[e11, e12], [e21, e22]])

    K_top = np.concatenate((K11, K12), axis=1)
    K_bot = np.concatenate((K21, K22), axis=1)

    K = np.concatenate((K_top, K_bot), axis=0)
    return K

K1 = s_m(theta, E, A, L)
K2 = s_m(theta2, E, A, L2)
# print(K1)
# print(K2)

K_global = np.zeros((6, 6))
K_global[0:4, 0:4] += K1
K_global[2:, 2:] += K2

# print(K_global)
# print(np.linalg.det(K_global))

# constraints
# K_global[0, :] = 0
# K_global[1, :] = 0
# K_global[4, :] = 0
# K_global[5, :] = 0

K_global[:, 0] = 0
K_global[:, 1] = 0
K_global[:, 4] = 0
K_global[:, 5] = 0

print(K_global)
print(np.linalg.det(K_global))

if __name__ == "__main__":

    pass