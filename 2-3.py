import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress = True) 
np.set_printoptions(precision = 6)

# Pauli Matrix
P = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
name = ['x', 'y', 'z']

# Calculate exp(i*theta/2*P)*v_0
def calcv (k, _theta):
    return np.dot(np.cos(_theta/2)*np.eye(2) + 1j*np.sin(_theta/2)*P[k], [[1.],[0,]])

def calc (i, j, _theta):
    v_theta = calcv (int(i), _theta)
    v_theta_dagger = np.conjugate(v_theta.T)
    return np.dot(v_theta_dagger, np.dot(P[int(j)], v_theta))

plt.figure(figsize = (12, 12), dpi = 80)

X = np.linspace(-2*np.pi, 2*np.pi, 256, dtype = complex)
for j in range(3):
    plt.subplot(3, 1, j+1)
    for k in range(3):
        Y = np.ones(256, dtype = complex)
        for i in range(256):
            Y[i] = calc(j, k, X[i])[0][0]
        plt.plot(X, Y, linewidth = 2.5,
                 label = r'$\hat{P} = \sigma_{%s}, \hat{Q} = \sigma_{%s}$' % (name[j], name[k]))
    plt.legend(loc = "upper left")
plt.show()