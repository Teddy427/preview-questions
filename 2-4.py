import numpy as np

P = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
I = np.array([[1, 0], [0, 1]])

n = int(input("input n:"))

H = np.zeros((2**n, 2**n), dtype = complex)

for i in range(n):
    R = np.array([[1]])
    for j in range(n):
        if i == j:
            R = np.kron(R, P[2])
        else:
            R = np.kron(R,I)
    H = H + R

for i in range(n-1):
    R = np.array([[1]])
    for j in range(n):
        if j == i:
            R = np.kron(R, P[0])
        elif j == i+1 :
            R = np.kron(R, P[0])
        else:
            R = np.kron(R, I)
    H = H + R

print(H)

v = np.zeros((2**n, 1), dtype = complex)
v[0][0] = 1.
print(np.dot(v.T, np.dot(H, v))[0][0])