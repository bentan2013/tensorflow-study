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

train_count = 10000
valid_count = 10000
test_count = 10000
image_size = 32
channels = 3
filter_size = 3
out_channels = channels
num_hidden_inc = 100
batch_size = 128
classes_number = 10
learing_rate = 0.1
num_steps = 10000

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



def get_data_set(name="train", cifar=10):
    x = None
    y = None
    l = None


    folder_name = "cifar-10-batches-py"

    f = open('d:/data/'+folder_name+'/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    l = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open('d:/data/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
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
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
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
train_src = train_src.reshape([-1, image_size, image_size, channels])

#print(np.shape(train_src))
#print(np.shape(train_target))


def accuracy(predictions, lables):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(lables, 1)) / predictions.shape[0]


# Create model
def conv_layer(input_data, inc, ouc):
    filter = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, inc, ouc), dtype=tf.float32))
    strides = [1, 1, 1, 1]
    conv = tf.nn.conv2d(input_data, filter=filter, strides=strides, padding='SAME')
    print("Shape of conv is{}".format(conv.get_shape()))
    bias = tf.Variable(tf.zeros(ouc, dtype=tf.float32))
    bias_add = tf.add(conv, bias)
    relu = tf.nn.relu(bias_add)
    ksize = [1,3,3,1]
    max_pool = tf.nn.max_pool(relu, ksize=ksize, strides=[1,1,1,1], padding="SAME")
    norm1 = tf.nn.lrn(max_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    return norm1


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
    rlt_lyr1 = conv_layer(input, 3, 64)
    rlt_lyr2 = conv_layer(rlt_lyr1, 64, 64)
    shape_after_conv = rlt_lyr2.get_shape().as_list()
    print(shape_after_conv)
    pixels_size = shape_after_conv[1] * shape_after_conv[2] * shape_after_conv[3]
    reshape = tf.reshape(rlt_lyr2, [-1, pixels_size])
    rlt_lyr4 = fc_layer(reshape, pixels_size, 384)
    rlt_lyr5 = fc_layer(rlt_lyr4, 384, 192)
    finally_rlt = fc_layer(rlt_lyr5, 192, classes_number)
    return finally_rlt


with tf.Session() as sess:
    tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, channels])
    tf_train_lables = tf.placeholder(dtype=tf.float32, shape=[None, classes_number])
    logits = nn_model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_lables, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(loss)
    train_prediction = tf.nn.softmax(nn_model(tf_train_dataset))
    tf.global_variables_initializer().run()

    for i in range(num_steps):
        offset = (i * batch_size) % (train_count - batch_size)
        batch_data = train_src[offset:(offset + batch_size), ...]
        batch_labels = train_labels[offset:(offset + batch_size), ...]

        _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict={
            tf_train_dataset:batch_data,
            tf_train_lables:batch_labels} )

        if i % 10 == 0:
            print("Current step is {0}, Loss is {1}, accuary is {2}%".format(i, l, accuracy(predictions, batch_labels)))

    writer = tf.summary.FileWriter("/tmp/cat_dog_logs", sess.graph)