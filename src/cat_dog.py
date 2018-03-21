#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np
import pickle
from scipy import ndimage
from scipy.misc import imresize, imsave


data_folder = "d:\\data\\cifar-10-batches-py\\"

# http://www.cs.toronto.edu/~kriz/cifar.html

# data -- a 10000x3072 numpy array of uint8s.
# Each row of the array stores a 32x32 colour image.
# The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
# The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

# Be familiar with CIFAR-10 dataset

label_name = (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck')

train_count = 40000
valid_count = 10000
test_count = 10000
image_size = 32
channels = 3
filter_size = 5
out_channels = channels
num_hidden_inc = 32
batch_size = 50
classes_number = 10
learing_rate = 0.01


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


def get_src_and_target_from_pickle(file):
    with open(data_folder + file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        labels = dict[b'labels']
        data = np.array(dict[b'data']).astype(np.float32) / 255.0
        count = len(labels)

        dataset_src = np.zeros((count, image_size, image_size, channels))
        dataset_target = np.zeros(shape=(count, classes_number))

        for i in range(count):
            rgb = np.zeros((32, 32, 3), dtype=np.float32)
            rgb[..., 0] = np.array(data[i][:1024]).reshape((32,32))
            rgb[..., 1] = np.array(data[i][1024:2048]).reshape((32,32))
            rgb[..., 2] = np.array(data[i][2048:3072]).reshape((32,32))
            dataset_src[i, ...] = rgb
            dataset_target[i][labels[i]] = 1.0

        return dataset_src, dataset_target


#test_unpickle(data_folder + "test_batch")


# Generate Train dataset and Test dataset
def generate_dataset(train_files, valid_files, test_files):
    train_src = np.zeros((0, image_size, image_size, channels), dtype=np.float32)
    valid_src = np.zeros((0, image_size, image_size, channels), dtype=np.float32)
    test_src = np.zeros((0, image_size, image_size, channels), dtype=np.float32)

    train_target = np.zeros((0, classes_number), dtype=np.float32)
    valid_target = np.zeros((valid_count, ), dtype=np.float32)
    test_target = np.zeros((test_count, ), dtype=np.float32)
    for t in train_files:
        src, target = get_src_and_target_from_pickle(t)
        train_src = np.append(train_src, src, axis=0)
        train_target = np.append(train_target, target, axis=0)

#    for t in valid_files:
#        src, target = get_src_and_target_from_pickle(t)
#        np.append(valid_src, src, axis=0)
#        np.append(valid_target, target, axis=0)

#    for t in test_files:
#        src, target = get_src_and_target_from_pickle(t)
#        np.append(test_src, src, axis=0)
#        np.append(test_target, target, axis=0)

    return train_src, train_target, valid_src, valid_target, test_src, test_target


train_src, train_target, valid_src, valid_target, test_src, test_target = generate_dataset(
    ["data_batch_1"],
    ["data_batch_4"],
    ["data_batch_5"]
)

#print(np.shape(train_src))
#print(np.shape(train_target))


def accuracy(predictions, lables):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(lables, 1)) / predictions.shape[0]


# Create model
def conv_layer(input_data):
    filter = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, channels, out_channels), dtype=tf.float32))
    strides = [1, 1, 1, 1]
    conv = tf.nn.conv2d(input_data, filter=filter, strides=strides, padding='SAME')
    print("Shape of conv is{}".format(conv.get_shape()))
    bias = tf.Variable(tf.zeros(channels, dtype=tf.float32))
    bias_add = tf.add(conv, bias)
    relu = tf.nn.relu(bias_add)
    ksize = [1,2,2,1]
    max_pool = tf.nn.max_pool(relu, ksize=ksize, strides=[1,1,1,1], padding="VALID")
    return max_pool


def fc_layer(input_data, input_size, output_size):
    print(input_size)
    print(output_size)
    weights = tf.truncated_normal(shape=(input_size, output_size), dtype=tf.float32)
    bias = tf.zeros(shape=(output_size), dtype=tf.float32)
    matmul = tf.matmul(input_data, weights)
    bias_add = tf.nn.bias_add(matmul, bias)
    relu = tf.nn.relu(bias_add)
    return relu


def nn_model(input):
    rlt_lyr1 = conv_layer(input)
    rlt_lyr2 = conv_layer(rlt_lyr1)
    rlt_lyr3 = conv_layer(rlt_lyr2)
    shape_after_conv = rlt_lyr3.get_shape().as_list()
    print(shape_after_conv)
    pixels_size = shape_after_conv[1] * shape_after_conv[2] * shape_after_conv[3]
    reshape = tf.reshape(rlt_lyr3, [shape_after_conv[0], pixels_size])
    rlt_lyr4 = fc_layer(reshape, pixels_size, num_hidden_inc)
    finally_rlt = fc_layer(rlt_lyr4, num_hidden_inc, classes_number)
    return finally_rlt


with tf.Session() as sess:
    tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=(batch_size, image_size, image_size, channels))
    tf_train_lables = tf.placeholder(dtype=tf.float32, shape=(batch_size, classes_number))

#    tf_valid_dataset = tf.constant(valid_src)
#    tf_valid_lables = tf.constant(valid_target)

#    tf_test_dataset = tf.placeholder(dtype=tf.float32, shape=(test_count, image_size, image_size, channels))
#    tf_test_lables = tf.placeholder(dtype=tf.float32, shape=(test_count, classes_number))

    logits = nn_model(tf_train_dataset)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_lables, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(loss)

    train_prediction = tf.nn.softmax(nn_model(tf_train_dataset))
    #vaild_prediction = tf.nn.softmax(nn_model(tf_train_dataset))
    #train_prediction = tf.nn.softmax(nn_model(tf_train_dataset))
    tf.global_variables_initializer().run()

    steps = train_count /4  / batch_size
    print("Total steps is {}".format(steps))
    for s in range(int(steps)):
        offset = s * batch_size
        batch_data = train_src[offset:(offset + batch_size), ...]
        batch_labels = train_target[offset:(offset + batch_size), ...]

        _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict={
            tf_train_dataset:batch_data,
            tf_train_lables:batch_labels} )

        if s % 10 == 0:
            print("Current step is {0}, Loss is {1}, accuary is {2}%".format(s, l, accuracy(predictions, batch_labels)))

    writer = tf.summary.FileWriter("/tmp/cat_dog_logs", sess.graph)











