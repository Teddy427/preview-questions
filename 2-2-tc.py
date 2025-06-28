import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

_theta = 1.

Rx = tc.gates.matrix_for_gate(tc.gates.rx_gate(theta = -2*_theta))
print(Rx)
_Rx = np.cos(_theta) * tc.gates._i_matrix + 1j * np.sin(_theta) * tc.gates._x_matrix
print(_Rx)

Ry = tc.gates.matrix_for_gate(tc.gates.ry_gate(theta = -2*_theta))
print(Ry)
_Ry = np.cos(_theta) * tc.gates._i_matrix + 1j * np.sin(_theta) * tc.gates._y_matrix
print(_Ry)

Rz = tc.gates.matrix_for_gate(tc.gates.rz_gate(theta = -2*_theta))
print(Rz)
_Rz = np.cos(_theta) * tc.gates._i_matrix + 1j * np.sin(_theta) * tc.gates._z_matrix
print(_Rz)