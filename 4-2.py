import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")
key = K.get_random_state(42)


def measure_on(key):
    K.set_random_state(key)
    c = tc.Circuit(2)
    c.h(0)
    c.cx(0, 1)
    return c.measure(0, 1)[0]

measure_on_jit = K.jit(measure_on, static_argnums=1)

ans = 0.
N = 1000
key1 = key
for _ in range(N):
    key1, key2 = K.random_split(key1)
    res = measure_on_jit(key2)
    cnt = 0
    for val in np.array(res):
        cnt = cnt + val
    ans = ans + (-1.) ** cnt
ans = ans / N
print(ans)