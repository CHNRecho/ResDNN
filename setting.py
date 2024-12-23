# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:45:44 2024

@author: DELL
"""

import tensorflow as tf

# Learning rate decay
def scheduler(epoch):
    # The learning rate remains unchanged in the first three epochs, and then decays proportionally after three epochs.
    if epoch < 3:
        return 0.0001
    else:
        lr = 0.0001 * tf.math.exp(0.08 * (3 - epoch))
        return lr.numpy()