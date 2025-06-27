import numpy as np
np.set_printoptions(suppress = True) 
np.set_printoptions(precision = 6)

# Pauli Matrix
P = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
theta = 1.

# Calculate exp(i*theta*P)
def expPauli (k, _theta):
    eig, Q = np.linalg.eig(_theta * 1j * P[k])
    expD = np.diag(np.exp(eig))
    return np.dot(Q, np.dot(expD, np.linalg.inv(Q)))

for i in range(3):
    print(expPauli(i, theta))
    print(np.cos(theta)*np.eye(2,2) + np.sin(theta) * 1j * P[i])