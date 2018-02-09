import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activate_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    output = tf.matmul(inputs, Weights) + biases
    if activate_function is None:
        return output
    else:
        return activate_function(output)


in_num = 100
out_num = 10
x_data = np.linspace(-1, 1, in_num, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# l1 = add_layer(

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100):
    print(sess.run(output_data))
