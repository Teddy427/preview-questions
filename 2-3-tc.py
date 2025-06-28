import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

name = ['i', 'x', 'y', 'z']

def calc (i, j, _theta):
    c = tc.Circuit(1)
    if i == 1:
        c.rx(0, theta = -_theta)
    elif i == 2:
        c.ry(0, theta = -_theta)
    else:
        c.rz(0, theta = -_theta)
    return c.expectation([tc.gates.matrix_for_gate(tc.gates.pauli_gates[j]), [0]])

X = np.linspace(-2*np.pi, 2*np.pi, 256)
Y = np.empty((256), dtype = complex)
plt.figure(figsize = (12, 8), dpi = 80)

for i in range(1,4):
    for j in range(1,4):
        plt.subplot(3, 3, 3*i+j-3)
        plt.xlim(-2.2*np.pi, 2.2*np.pi)
        plt.ylim(-1.1, 1.1)
        plt.xticks(np.linspace(-2*np.pi, 2*np.pi, 5))
        for k in range(256):
            Y[k] = calc(i, j, X[k])
        plt.plot(X, Y, linewidth = 2.5,
                 label = r'$\hat{P} = \sigma_{%s}, \hat{Q} = \sigma_{%s}$' % (name[i], name[j]))
        plt.legend(loc = "upper left")
plt.show()