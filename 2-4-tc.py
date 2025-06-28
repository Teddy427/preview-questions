import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

n = int(input("input n:"))

H = np.zeros((2**n, 2**n), dtype = complex)

for i in range(n):
    R = np.array([[1]], dtype = complex)
    for j in range(n):
        if i == j:
            R = tc.backend.kron(R, tc.gates._z_matrix + 0j)
        else:
            R = tc.backend.kron(R, tc.gates._i_matrix + 0j)
    H = H + R

for i in range(n-1):
    R = np.array([[1]], dtype = complex)
    for j in range(n):
        if j == i:
            R = tc.backend.kron(R, tc.gates._x_matrix + 0j)
        elif j == i+1 :
            R = tc.backend.kron(R, tc.gates._x_matrix + 0j)
        else:
            R = tc.backend.kron(R, tc.gates._i_matrix + 0j)
    H = H + R

print(np.array(H))

c = tc.Circuit(n)

print(np.array(c.expectation([H, np.arange(n)])))