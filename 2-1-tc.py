import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

_theta = np.pi / 3

c = tc.Circuit(1)

plt.plot([0,c.amplitude("0")], [0,c.amplitude("1")])

c.ry(0, theta = -2*_theta)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.plot([0,c.amplitude("0")], [0,c.amplitude("1")])