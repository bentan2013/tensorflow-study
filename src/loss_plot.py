#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt


input = tf.linspace(-1., 1., 500, "input")
output = tf.constant(0., name="output")




sess = tf.Session()

show_regression_loss = False

if show_regression_loss is True:
    # for regression
    l1_loss = tf.abs(output - input)
    l2_loss = tf.square(output - input)
    l1_loss_out = sess.run(l1_loss)
    l2_loss_out = sess.run(l2_loss)
    input_src = sess.run(input)
    plt.plot(input_src, l1_loss_out, 'b--', label='L1 Loss')
    plt.plot(input_src, l2_loss_out, 'r--', label='L2 Loss')
    plt.legend(loc='lower right')
    plt.show()
else:
    # for classification
    x_vals = tf.linspace(-3., 5., 500, "x_vals")
    target = tf.constant(1.)
    targets = tf.fill([500,], target)
    cross_entropy_loss = - tf.multiply(targets,
                                   (tf.log(x_vals)) - tf.multiply((1. -target),
                                                                  tf.log(1. -x_vals)))
    x_vals_src = sess.run(x_vals)
    cross_entropy_loss_out = sess.run(cross_entropy_loss)
    plt.plot(x_vals_src, cross_entropy_loss_out, 'g--', label='Cross entropy')
    plt.ylim(-1.5, 3)
    plt.legend(loc='lower right')
    plt.show()

