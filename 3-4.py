import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorcircuit as tc

K = tc.set_backend("tensorflow")

# Example : P1=X,P2=Y
def f (params):
    c = tc.Circuit(1)
    c.exp1(0, theta = -params[0],
          unitary = tc.gates._x_matrix)
    e = c.expectation([tc.gates.y(), [0]])
    return K.real(e)
    
f_val_grad = K.value_and_grad(f)
f_val_grad_jit = K.jit(f_val_grad)

learning_rate = 5e-3
opt = K.optimizer(tf.keras.optimizers.SGD(learning_rate))


def grad_descent(params):
    val, grad = f_val_grad_jit(params)
    params = opt.update(grad, params)
    return params

params = K.implicit_randn(1)
for i in range(500):
    params = grad_descent(params)
print("The minimum is %.3f" % np.array(f(params)))