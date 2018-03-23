#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class_1_data = np.random.normal(-1, 0.5, 60)
class_2_data = np.random.normal(3, 0.5, 60)


train_data = np.concatenate((class_1_data[:50], class_2_data[:50]))
train_data = np.reshape(train_data, newshape=(100,1))
train_labels = np.concatenate((np.repeat(0, 50 ), np.repeat(1, 50)))
labels_r2 = np.zeros(shape=(100, 2), dtype=np.float32)

test_data = np.reshape(np.concatenate((class_1_data[50:], class_2_data[50:])), newshape=(20, 1))

test_labels = np.concatenate((np.repeat(0, 10), np.repeat(1, 10)))

test_r2 = np.zeros(shape=(20, 2), dtype=np.float32)


for i in range(100):
    labels_r2[i][train_labels[i]] = 1.0

for i in range(20):
    test_r2[i][test_labels[i]] = 1.0


steps = 100000
batch_size = 100

with tf.Session() as sess:
    xs = tf.placeholder(dtype=tf.float32, shape=(batch_size, 1))
    ys = tf.placeholder(dtype=tf.float32, shape=(batch_size, 2))
    weight = tf.Variable(tf.random_normal(shape=[1, 2], mean=10))
    bias = tf.Variable(tf.random_normal(shape=[1], mean=10))

    out = tf.add(tf.matmul(xs, weight), bias)

    loss = tf.abs(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=ys)))
    opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    prediction = tf.nn.softmax(out)

    sess.run(tf.global_variables_initializer())

    for i in range(steps):
        randx = np.zeros(shape=[batch_size, 1])
        randy = np.zeros(shape=[batch_size, 2])

        testx = np.zeros(shape=[batch_size, 1])
        testy = np.zeros(shape=[batch_size, 2])

        for j in range(batch_size):
            ri = np.random.choice(100)
            ti = np.random.choice(20)
            randx[j] = train_data[ri]
            randy[j] = labels_r2[ri]
            testx[j] = test_data[ti]
            testy[j] = test_r2[ti]
        _, l, b = sess.run([opt, loss, bias], feed_dict={xs:randx, ys:randy})
        if i % 100 == 0:
           print("Loss is {0}, A is {1}".format(l, b))
        if i % 1000 == 0:
            predic, labels, w, b = sess.run([prediction, ys, weight, bias], feed_dict={xs:testx, ys:testy})
            acc = 100.0 * np.sum(np.argmax(predic, 1) == np.argmax(labels, 1)) / predic.shape[0]
            print("A is {0}, Accuracy is {1}%".format(b, acc))
            print(w)
            print(b)







