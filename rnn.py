#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Müller / mmueller@cl.uzh.ch

import numpy as np

from typing import Sequence

from activations import sigmoid, tanh


class RNN:

    def __init__(self) -> None:

        self.params = {}

    def step(self, input_):
        raise NotImplementedError

    def output(self):
        """
        At a certain point in time, compute an output.
        """
        raise NotImplementedError

    def forward_sequence(self, sequence):
        raise NotImplementedError


class ElmanScalarRNN(RNN):
    """
    An Elman or "vanilla" RNN that, at each time step,
    takes scalar input.

    >>> e = ElmanScalarRNN(10)

    >>> print(e.step(5))
    [0.48501544 0.49107564 0.504578   0.50550117 0.51805654 0.48205011
     0.51270459 0.47850066 0.49718829 0.50798906]
    >>> print(e.output())
    [-0.00388719]
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        # in this scalar input version, the weight matrices are
        # vectors, and the biases are scalars
        self.params["Wh"] = np.random.randn(hidden_size,) * 0.01
        self.params["Uh"] = np.random.randn(hidden_size,) * 0.01
        self.params["bh"] = np.zeros(1,)

        self.params["Wy"] = np.random.randn(hidden_size,) * 0.01
        self.params["by"] = np.zeros(1,)

        self.reset_state()

    def reset_state(self) -> None:

        self.hidden_state = 0.0

    def step(self, input_) -> np.ndarray:

        input_linear = np.dot(self.params["Wh"], input_)
        hidden_linear = np.dot(self.params["Uh"], self.hidden_state) + self.params["bh"]

        # compute new hidden state
        self.hidden_state = sigmoid(input_linear + hidden_linear)

        return self.hidden_state

    def forward_sequence(self, sequence: Sequence[np.ndarray]) -> Sequence[np.ndarray]:

        self.reset_state()

        hidden_states = []

        for item in sequence:
            hidden_states.append(self.step(item))

        return hidden_states

    def output(self) -> np.ndarray:
        return np.dot(self.params["Wy"], self.hidden_state) + self.params["by"]


class ElmanRNN(RNN):
    """
    An Elman RNN that, at each time step,
    takes vector input.

    >>> e = ElmanRNN(5, 10, 5)
    >>> print(e.step(np.array([1, 2, 3, 4, 5])))
    [0.53239282 0.50239672 0.50754092 0.49467159 0.47343499 0.48537873
     0.48764785 0.50135803 0.51225634 0.49526006]
    >>> print(e.output())
    [ 0.00053832  0.00782895 -0.01356998  0.01432975 -0.03725151]
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # in this vector version, weight matrices are actual
        # matrices, and biases are vectors
        self.params["Wh"] = np.random.randn(input_size, hidden_size) * 0.01
        self.params["Uh"] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.params["bh"] = np.zeros(hidden_size,)

        self.params["Wy"] = np.random.randn(hidden_size, output_size) * 0.01
        self.params["by"] = np.zeros(output_size,)

        self.reset_state()

    def reset_state(self) -> None:

        self.hidden_state = np.zeros(self.hidden_size,)

    def step(self, input_: np.ndarray) -> np.ndarray:

        input_linear = np.dot(input_, self.params["Wh"])
        hidden_linear = np.dot(self.params["Uh"], self.hidden_state) + self.params["bh"]

        # compute new hidden state
        self.hidden_state = sigmoid(input_linear + hidden_linear)

        return self.hidden_state

    def forward_sequence(self, sequence: Sequence[np.ndarray]) -> Sequence[np.ndarray]:

        self.reset_state()

        hidden_states = []

        for item in sequence:
            hidden_states.append(self.step(item))

        return hidden_states

    def output(self) -> None:
        return np.dot(self.hidden_state, self.params["Wy"]) + self.params["by"]


class LSTM(RNN):
    """
    Long-short term memory RNN, vanilla without peephole
    connections.

    >>> e = LSTM(5, 10, 5)
    >>> print(e.step(np.array([1, 2, 3, 4, 5])))
    [ 9.81669866e-05  3.49891068e-03  2.03026849e-02  3.67467230e-03
    -5.13049272e-03 -4.68873686e-03  8.13246236e-03  5.22321287e-03
    2.76350076e-03 -4.19133057e-03]
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # forget gate parameters
        self.params["Wf"] = np.random.randn(hidden_size, input_size) * 0.01
        self.params["Uf"] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.params["bf"] = np.zeros(hidden_size,)

        # input gate parameters
        self.params["Wi"] = np.random.randn(hidden_size, input_size) * 0.01
        self.params["Ui"] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.params["bi"] = np.zeros(hidden_size,)

        # output gate parameters
        self.params["Wo"] = np.random.randn(hidden_size, input_size) * 0.01
        self.params["Uo"] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.params["bo"] = np.zeros(hidden_size,)

        # cell state parameters
        self.params["Wc"] = np.random.randn(hidden_size, input_size) * 0.01
        self.params["Uc"] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.params["bc"] = np.zeros(hidden_size,)

        # output vector parameters
        self.params["Wy"] = np.random.randn(hidden_size, output_size) * 0.01
        self.params["by"] = np.zeros(output_size,)

        self.reset_state()

    def reset_state(self) -> None:

        self.hidden_state = np.zeros(self.hidden_size,)
        self.cell_state = np.zeros(self.hidden_size,)

    def step(self, input_: np.ndarray) -> np.ndarray:

        # forget gate
        forget_gate_input = np.dot(self.params["Wf"], input_)
        forget_gate_hidden = np.dot(self.params["Uf"], self.hidden_state)
        forget_gate = sigmoid(forget_gate_input + forget_gate_hidden + self.params["bf"])

        # input gate
        input_gate_input = np.dot(self.params["Wi"], input_)
        input_gate_hidden = np.dot(self.params["Ui"], self.hidden_state)
        input_gate = sigmoid(input_gate_input + input_gate_hidden + self.params["bi"])

        # output gate
        output_gate_input = np.dot(self.params["Wo"], input_)
        output_gate_hidden = np.dot(self.params["Uo"], self.hidden_state)
        output_gate = sigmoid(output_gate_input + output_gate_hidden + self.params["bo"])

        # cell state
        cell_state_input = np.dot(self.params["Wc"], input_)
        cell_state_hidden = np.dot(self.params["Uc"], self.hidden_state)
        cell_state = tanh(cell_state_input + cell_state_hidden + self.params["bc"])

        self.cell_state = (forget_gate * self.cell_state) + (input_gate * cell_state)

        self.hidden_state = output_gate * tanh(self.cell_state)

        return self.hidden_state

    def forward_sequence(self, sequence: Sequence[np.ndarray]) -> Sequence[np.ndarray]:

        self.reset_state()

        hidden_states = []

        for item in sequence:
            hidden_states.append(self.step(item))

        return hidden_states

    def output(self) -> None:
        return np.dot(self.hidden_state, self.params["Wy"]) + self.params["by"]
