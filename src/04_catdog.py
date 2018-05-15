#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle
from scipy.misc import imsave


data_folder = "../data/cifar-10-batches-py/"

# http://www.cs.toronto.edu/~kriz/cifar.html

# data -- a 10000x3072 numpy array of uint8s.
# Each row of the array stores a 32x32 colour image.
# The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
# The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

# Be familiar with CIFAR-10 dataset
label_name = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']

train_count = 10000
valid_count = 10000
test_count = 10000
image_size = 32
channels = 3
filter_size = 3
out_channels = 3
num_hidden_inc = 32
batch_size = 128
classes_number = 10
learning_rate = 0.1
stddev = 0.1
SEED = 11215
dropout_prob = 0.25
num_steps = 10000


def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    _RESHAPE_SIZE = 4*4*128

    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

    def variable_with_weight_decay(name, shape, stddev, wd):
        dtype = tf.float32
        var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    with tf.variable_scope('conv4') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv4', conv4)
    tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

    with tf.variable_scope('conv5') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv5', conv5)
    tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('fully_connected1') as scope:
        reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('fully_connected2') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc2', local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    with tf.variable_scope('output') as scope:
        weights = variable_with_weight_decay('weights', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [_NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(softmax_linear, axis=1)

    return x, y, softmax_linear, global_step, y_pred_cls


def test_unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        labels = dict[b'labels']
        count = len(labels)
        index = 2
        name = label_name[dict[b'labels'][index]] + '.png'
        img_data = dict[b'data'][index]
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[..., 0] = np.array(img_data[:1024]).reshape((32,32))
        rgb[..., 1] = np.array(img_data[1024:2048]).reshape((32,32))
        rgb[..., 2] = np.array(img_data[2048:3072]).reshape((32,32))
        imsave(name, rgb)
        data = dict[b'data']
        dataset_src = np.zeros((count, image_size, image_size, channels))
        for i in range(count):
            rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            rgb[..., 0] = np.array(data[i][:1024]).reshape((32,32))
            rgb[..., 1] = np.array(data[i][1024:2048]).reshape((32,32))
            rgb[..., 2] = np.array(data[i][2048:3072]).reshape((32,32))
            dataset_src[i, ...] = rgb
        imsave("test_reshape.png", dataset_src[0, ...])


def get_data_set(name="train", cifar=10):
    x = None
    y = None
    l = None

    f = open(data_folder + 'batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    l = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open(data_folder + 'data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open(data_folder + 'test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    return x, dense_to_one_hot(y), l


train_src, train_labels, l = get_data_set()


with tf.Session() as sess:
    x, y, output, global_step, y_pred_cls = model()
    print(x.get_shape())
    print(y.get_shape())

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("Accuracy/train", accuracy)
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("../tmp/04_catdog", sess.graph)
    tf.global_variables_initializer().run()

    for i in range(num_steps):
        offset = (i * batch_size) % (train_count - batch_size)

        batch_xs = train_src[offset:(offset + batch_size), ...]
        batch_ys = train_labels[offset:(offset + batch_size), ...]

        i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})

        if (i_global % 10 == 0) or (i == num_steps - 1):
            _loss, batch_acc = sess.run([loss, accuracy], feed_dict={x: batch_xs, y: batch_ys})
            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f})"
            print(msg.format(i_global, batch_acc, _loss))

#       summary_result, _, l, relu4_rlt, conv1_rlt, relu1_rlt, max_pool1_rlt, reshape_rlt, train_prediction_rlt = sess.run([merged, optimizer, loss, relu4, conv1, relu1, max_pool1, reshape, train_prediction], feed_dict={
#          tf_train_dataset:batch_data,
#           tf_train_lables:batch_labels})

#       if s % 10 == 0:
#           print("Current step is {0}, Loss is {1}, accuracy is {2}".format(s, l, accuracy(train_prediction_rlt, batch_labels)))

#       if s < 3:
#           for i in range(1):
#               index = np.argmax(batch_labels[i]).astype(np.uint8)
#               p_index = np.argmax(train_prediction_rlt[i]).astype(np.uint8)
#               imsave("./data/{0}_{1}_{2}_{3}.png".format(s, i, label_name[index], label_name[p_index]), batch_data[i])
#               imsave("./data/{0}_{1}_{2}_{3}_conv1.png".format(s, i, label_name[index], label_name[p_index]), max_pool1_rlt[i])
#               np.savetxt("./data/{0}_{1}_{2}_{3}_reshape_rlt.txt".format(s, i, label_name[index], label_name[p_index]), reshape_rlt, fmt='%.4f')
#               np.savetxt("./data/{0}_{1}_{2}_{3}_prediction.txt".format(s, i, label_name[index], label_name[p_index]), train_prediction_rlt, fmt='%.4f')

#       if s >= 2000 and s % 500 == 0:
#           index = np.argmax(batch_labels[0]).astype(np.uint8)
#           p_index = np.argmax(train_prediction_rlt[0]).astype(np.uint8)
#           imsave("./data/{0}_{1}_{2}.png".format(s, label_name[index], label_name[p_index]), batch_data[0])
##          imsave("./data/{0}_{1}_{2}_conv1.png".format(s, label_name[index], label_name[p_index]), conv1_rlt[0])
#           imsave("./data/{0}_{1}_{2}_relu1.png".format(s, label_name[index], label_name[p_index]), relu1_rlt[0])
#           imsave("./data/{0}_{1}_{2}_pool1.png".format(s, label_name[index], label_name[p_index]), max_pool1_rlt[0])
#           np.savetxt("./data/{0}_{1}_{2}_{3}_reshape_rlt.txt".format(s, i, label_name[index], label_name[p_index]),
#                  reshape_rlt, fmt='%.4f')
#           np.savetxt("./data/{0}_{1}_{2}_{3}_prediction.txt".format(s, i, label_name[index], label_name[p_index]),
#                  train_prediction_rlt, fmt='%.4f')
#            imsave("{0}_{1}_{2}_conv2.png".format(s, label_name[index], label_name[p_index]), max_pool1_rlt[0])
#            imsave("{0}_{1}_{2}_conv3.png".format(s, label_name[index], label_name[p_index]), conv3_rlt[0])
#       writer.add_summary(summary_result, s)

