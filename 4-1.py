import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)

e = c.expectation_ps(z = [0, 1])
print("%.6f" % K.real(np.array(e)))