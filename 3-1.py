import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

def f (array):
    sum = 0.
    for val in array:
        sum = sum + val**3
    return sum

X = np.array([11., 45., 14.])
n = len(X)
grad = np.zeros((n))
eps = 1e-6

for i in range(n):
    _x = np.zeros((n))
    _x[i] = eps
    grad[i] = (f(X+_x) - f(X)) / eps
print(grad)