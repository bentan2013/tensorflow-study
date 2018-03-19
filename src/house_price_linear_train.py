#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

learning_rate = 0.1
record_count = 0
x_center = 116.407718
y_center = 38.915599

max = 154953.0
count = 5884

dis_np = np.zeros((count, 1), dtype=np.float32)
price_np = np.zeros((count, 1), dtype=np.float32)
info_np = np.zeros((count, 2), dtype=np.float32)

index = 0
csv_file = "../data/house_price_number.csv"

with open(csv_file, 'rt', encoding='utf-8') as ifile:
    reader = csv.DictReader(ifile)
    rows = [row for row in reader]

    for row in rows:
        dis = float(row['distance'])
        price = float(row['price']) / max
        dis_np[index] = [dis]
        price_np[index] = [price]
        index += 1


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


def add_layer(inputs, in_size, out_size, activation_funcation = None):

    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    print(Weights.name, biases.name)

    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

    if activation_funcation is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_funcation(Wx_plus_b)

    return outputs

print(dis_np.shape)
print(price_np.shape)
print(info_np.shape)


l1 = add_layer(xs, 1, 10, activation_funcation=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_funcation=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

graph = tf.get_default_graph()
weights_1 = graph.get_tensor_by_name("Variable:0")
biases_1 = graph.get_tensor_by_name("Variable_1:0")
weights_2 = graph.get_tensor_by_name("Variable_2:0")
biases_2 = graph.get_tensor_by_name("Variable_3:0")
tf.summary.histogram('weights_1', weights_1)
tf.summary.histogram('biases_1', biases_1)
tf.summary.histogram('weights_2', weights_2)
tf.summary.histogram('biases_2', biases_2)
tf.summary.scalar("loss", loss)

writer = tf.summary.FileWriter("/tmp/tf_linear_logs", sess.graph)
merged = tf.summary.merge_all()

# tensorboard --logdir="/tmp/tf_linear_logs" and you can see the dashboard


for i in range(100):
    sum_string, step, l= sess.run([merged, train_step, loss], feed_dict={xs:dis_np, ys:price_np})
    writer.add_summary(sum_string, i)
    if i % 5 == 0:
        print("loss:{}".format(l))

res_prediction = sess.run(prediction, feed_dict={xs:dis_np, ys:price_np})

plt.plot(dis_np, price_np, 'ro', label='Original data')
plt.plot(dis_np, np.abs(res_prediction - price_np), label='Fitting Line')
plt.show()



