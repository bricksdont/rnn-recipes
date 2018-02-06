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



def dynamic_rnn():
    """
    A dynamic RNN that can take batches of varying size,
    i.e. what varies is the sequence length (= timesteps).
    """
    num_input = 5
    num_hidden = 10
    batch_size = 2
    timesteps = 10

    # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell
    lstm = rnn.LSTMCell(num_units=num_hidden)

    lstm_inputs = tf.placeholder("float", [None, None, num_input])

    # optional parameter, if supplied, states are zeroed out correctly
    sequence_lengths = tf.placeholder(tf.int32, [None])

    # https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    lstm_output = tf.nn.dynamic_rnn(cell=lstm,
                                  inputs=lstm_inputs,
                                  dtype=tf.float32,
                                  sequence_length=sequence_lengths)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        # prepare actual input data (batch_size, time_steps, num_input)
        actual_input = np.random.randn(batch_size, timesteps, num_input)

        # pad the second example in the batch so that it has length 5
        actual_input[1, 5:] = 0
        lengths = [10, 5]

        outputs, states = sess.run(lstm_output, feed_dict={lstm_inputs: actual_input,
                                                           sequence_lengths: lengths})

        print("outputs: ", outputs)
        print("states: ", states)

dynamic_rnn()
