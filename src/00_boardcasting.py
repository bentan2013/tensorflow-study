#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

train = np.random.rand(3, 2)
test = np.ones(shape=(3, 2) )
print("train:{0}\ttest:{1}".format(train, test))

test = np.expand_dims(test, 1)
print("New test:{0}".format(test))

sub = np.subtract(train, test)
print("shape:{0}\nresult:{1}".format(np.shape(sub), sub))

