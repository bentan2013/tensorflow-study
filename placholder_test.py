# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-5-placeholde/
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

sess = tf.Session()

print(sess.run(output, feed_dict={input1: [1], input2: [2]}))
