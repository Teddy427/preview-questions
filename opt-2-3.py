import numpy as np
import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")
n = (int)(input("input n: "))
x = np.zeros((n))
y = np.zeros((n))
for i in range(n):
    x[i], y[i] = map(float, (input().split()))
tx = tf.convert_to_tensor(x, dtype = tf.float32)
ty = tf.convert_to_tensor(y, dtype = tf.float32)

def f (params):
    # y = kx + b, kx - y + b = 0
    return tf.reduce_sum(tf.abs(params[0] * tx - ty + params[1]) / tf.sqrt(params[0] ** 2 + 1))

f_val_grad = K.value_and_grad(f)
f_val_grad_jit = K.jit(f_val_grad)

learning_rate = 2e-3
opt = K.optimizer(tf.keras.optimizers.SGD(learning_rate))

def grad_descent(params):
    val, grad = f_val_grad_jit(params)
    params = opt.update(grad, params)
    return params

sumy = 0
for i in range(n):
    sumy += y[i]
params = tf.Variable([0, sumy/n])
for i in range(1000):
    params = grad_descent(params)
print("The best line is y = %.2f x+ %.2f" % (K.numpy(params)[0], K.numpy(params)[1]))