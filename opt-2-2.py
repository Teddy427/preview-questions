import numpy as np
import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")
n = (int)(input("input n: "))
x = np.zeros((n))
y = np.zeros((n))
for i in range(n):
    x[i], y[i] = map(float, (input().split()))
tx = tf.convert_to_tensor(x)
ty = tf.convert_to_tensor(y)

def f (params):
    return tf.reduce_sum(tf.sqrt((tx - params[0]) ** 2 + (ty - params[1]) ** 2))

f_val_grad = K.value_and_grad(f)
f_val_grad_jit = K.jit(f_val_grad)

learning_rate = 2e-3
opt = K.optimizer(tf.keras.optimizers.SGD(learning_rate))

def grad_descent(params):
    val, grad = f_val_grad_jit(params)
    params = opt.update(grad, params)
    return params

sumx = 0
sumy = 0
for i in range(n):
    sumx += x[i]
    sumy += y[i]
params = tf.Variable([sumx/n, sumy/n])
for i in range(1000):
    params = grad_descent(params)
print("The best point is (%.2f, %.2f)" % (K.numpy(params)[0], K.numpy(params)[1]))