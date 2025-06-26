import numpy as np
import matplotlib.pyplot as plt
def Plot (v):
    x = np.array([0, v[0][0]])
    y = np.array([0, v[1][0]])
    plt.plot(x, y)
pi = np.pi
theta = pi / 3
v = [[1],[0]]
R = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
v0 = np.dot(R, v)
Plot(v)
Plot(v0)
plt.show()
# rotate the vector clockwise by theta