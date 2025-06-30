import numpy as np
import matplotlib.pyplot as plt
import tensorcircuit as tc
import cotengra
opt = cotengra.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    minimize="combo",
    max_time=12,
    max_repeats=1024,
    progbar=True,
)
tc.set_contractor("custom", optimizer=opt, contraction_info=True, preprocessing=True)
K = tc.set_backend("tensorflow")

'''
0 1
2 3
4 5
'''
adj = [[0,1,6],[0,2,7],[1,3,8],[2,3,9],[2,4,10],[3,5,11],[4,5,12]]

c = tc.Circuit(19)
n = 6
N = 2**6
M = 2
_theta = np.arcsin(np.sqrt(1. * M / N)) * 2.
offset = 2. * np.dot(np.eye(N, 1), np.eye(1, N)) - np.eye(N, N)

def oracle():
    global c
    for i1,i2,j in adj:
        c.cnot(i1, j)
        c.cnot(i2, j)
    c.toffoli(6, 7, 13)
    for i in range(8, 13):
        c.toffoli(i, i+5, i+6)
    # index 18 : the final result

def reoracle():
    global c
    for i in range(9, 13):
        c.toffoli(20-i, 25-i, 26-i)
    c.toffoli(6, 7, 13)
    for i1,i2,j in adj:
        c.cnot(i1,j)
        c.cnot(i2,j)

def grover():
    global c
    oracle()
    reoracle()
    for i in range(n):
        c.h(i)
    c.unitary(0, 1, 2, 3, 4, 5, unitary = offset)
    for i in range(n):
        c.h(i)
    
for i in range(n):
    c.H(i)
c.x(18)
c.h(18)
G = int(np.arccos(np.sqrt(1. * M / N)) / _theta + 0.5)
X = np.arange(G+2)
Y = np.zeros((G+2), dtype = float)
for i in range(G+2):
    c.h(18)
    Y[i] = 2. * c.amplitude("0110010000000000001") ** 2
    c.h(18)
    grover()
plt.plot(X, Y)
plt.scatter(X, Y, 50)
for x, y in zip(X, Y):
    plt.text(x+0.1, y-0.01, '%.3f' % y)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
'''
for _ in range(20):
    print(c.measure(0,1,2,3,4,5,with_prob=True))
'''