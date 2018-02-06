#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Müller / mmueller@cl.uzh.ch

import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn


def static_rnn():
    """
    Demonstrates a static tf lstm cell, i.e. where number of time
    steps is fixed.
    """

    timesteps = 2
    num_input = 5
    num_hidden = 10
    batch_size = 2

    # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell
    lstm = rnn.LSTMCell(num_units=num_hidden)

    lstm_inputs = tf.placeholder("float", [timesteps, batch_size, num_input])

    # unstack the (timesteps, batch_size, num_input) array to get list of
    # (batch_size, num_input) arrays
    lstm_inputs_list = tf.unstack(lstm_inputs, timesteps, axis=0)

    # https://www.tensorflow.org/api_docs/python/tf/nn/static_rnn
    lstm_output = rnn.static_rnn(lstm, lstm_inputs_list, dtype=tf.float32)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        # prepare actual input data (timesteps, batch_size, num_input)
        actual_input = np.random.randn(timesteps, batch_size, num_input)

        outputs, states = sess.run(lstm_output, feed_dict={lstm_inputs: actual_input})

        print("outputs: ", outputs)
        print("states: ", states)

static_rnn()


