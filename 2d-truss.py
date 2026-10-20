import numpy as np
from numpy import *
from scipy import *

E = [200e9,300e9]  #young's modulus
A = [10,20]        #Areas
# node coordinates: node 1 and node 2 fixed (base), node 3 free (top)
nodes = np.array([
    [0, 0],
    [2, 0],
    [1, 1]
])

members = [(0,2),(1,2)]

#individual stiffness K
# member 1: node 1 -> node 3
dx1, dy1 = nodes[2] - nodes[0]
L1 = np.hypot(dx1, dy1)
k1 = E[0] * A[0] / L1

# member 2: node 2 -> node 3
dx2, dy2 = nodes[2] - nodes[1]
L2 = np.hypot(dx2, dy2)
k2 = E[1] * A[1] / L2

K = np.zeros((6, 6),dtype=float)

for m, (i, j) in enumerate(members):
           dx, dy = nodes[j] - nodes[i]
           L = np.hypot(dx, dy)
           c, s = dx / L, dy / L
           k = np.array([[ c*c,  c*s, -c*c, -c*s],
                  [ c*s,  s*s, -c*s, -s*s],
                  [-c*c, -c*s,  c*c,  c*s],
                  [-c*s, -s*s,  c*s,  s*s]]) * E[m] * A[m] / L
           dofs = [2*i, 2*i+1, 2*j, 2*j+1]
           K[np.ix_(dofs, dofs)] += k


K = np.array(K, dtype=float)

print("Global K:\n", K)

# remove fixed DOFs (u1, v1, u2, v2) -> keep rows/columns 4 and 5 (u3, v3)
free = [4, 5]
K_red = K[np.ix_(free, free)]

print("Reduced K:\n", K_red)

# load at node 3: Fx = 10, Fy = -11
F = np.array([[10], [-11]])

d = np.linalg.solve(K_red, F)
print("Displacements of node 3 (u3, v3):\n", d)