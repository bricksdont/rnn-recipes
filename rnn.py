#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Müller / mmueller@cl.uzh.ch

import numpy as np

from typing import Sequence
from collections import namedtuple

from activations import sigmoid, tanh


class RNN:

    def __init__(self) -> None:

        self.params = {}

    def step(self, input_):
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
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        # in this scalar input version, the weight matrices are
        # vectors, and the biases are scalars
        self.params["Wh"] = np.random.randn(hidden_size,) * 0.01
        self.params["Uh"] = np.random.randn(hidden_size,) * 0.01
        self.params["bh"] = np.zeros(1,)

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


class ElmanRNN(RNN):
    """
    An Elman RNN that, at each time step,
    takes vector input.

    >>> e = ElmanRNN(5, 10)
    >>> print(e.step(np.array([1, 2, 3, 4, 5])))
    [0.53239282 0.50239672 0.50754092 0.49467159 0.47343499 0.48537873
     0.48764785 0.50135803 0.51225634 0.49526006]
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # in this vector version, weight matrices are actual
        # matrices, and biases are vectors
        self.params["Wh"] = np.random.randn(input_size, hidden_size) * 0.01
        self.params["Uh"] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.params["bh"] = np.zeros(hidden_size,)

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

LSTMState = namedtuple("LSTMState", ["cell_states", "hidden_states"])

class LSTM(RNN):
    """
    Long-short term memory RNN, vanilla without peephole
    connections.

    >>> e = LSTM(5, 10)
    >>> print(e.step(np.array([1, 2, 3, 4, 5])))
    [ 9.81669866e-05  3.49891068e-03  2.03026849e-02  3.67467230e-03
    -5.13049272e-03 -4.68873686e-03  8.13246236e-03  5.22321287e-03
    2.76350076e-03 -4.19133057e-03]
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int) -> None:
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

    def forward_sequence(self, sequence: Sequence[np.ndarray]) -> LSTMState:

        self.reset_state()

        hidden_states = []
        cell_states = []

        for item in sequence:
            hidden_state, cell_state = self.step(item)
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)

        return LSTMState(hidden_states, cell_states)

class GRU(RNN):
    """
    Gated recurrent unit RNN.

    >>> g = GRU(5, 10)
    >>> print(g.step(np.array([1, 2, 3, 4, 5])))

    [-0.02601452 -0.00179476 -0.01031278 -0.02597636 -0.02946662  0.02102406
     -0.04205244 -0.03658039  0.00968579  0.02418459]
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # update gate parameters
        self.params["Wz"] = np.random.randn(hidden_size, input_size) * 0.01
        self.params["Uz"] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.params["bz"] = np.zeros(hidden_size,)

        # reset gate parameters
        self.params["Wr"] = np.random.randn(hidden_size, input_size) * 0.01
        self.params["Ur"] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.params["br"] = np.zeros(hidden_size,)

        # hidden proposal parameters
        self.params["Wh"] = np.random.randn(hidden_size, input_size) * 0.01
        self.params["Uh"] = np.random.randn(hidden_size, hidden_size) * 0.01
        self.params["bh"] = np.zeros(hidden_size,)

        self.reset_state()

    def reset_state(self) -> None:

        self.hidden_state = np.zeros(self.hidden_size,)

    def step(self, input_: np.ndarray) -> np.ndarray:

        # update gate
        update_gate_input = np.dot(self.params["Wz"], input_)
        update_gate_hidden = np.dot(self.params["Uz"], self.hidden_state)
        update_gate = sigmoid(update_gate_input + update_gate_hidden + self.params["bz"])

        # reset gate
        reset_gate_input = np.dot(self.params["Wr"], input_)
        reset_gate_hidden = np.dot(self.params["Ur"], self.hidden_state)
        reset_gate = sigmoid(reset_gate_input + reset_gate_hidden + self.params["br"])

        # hidden state proposal
        proposal_input = np.dot(self.params["Wh"], input_)
        proposal_hidden = np.dot(self.params["Uh"], (self.hidden_state * reset_gate))
        proposal = tanh(proposal_input + proposal_hidden + self.params["bh"])

        # new hidden state
        self.hidden_state = ((1 - update_gate) * proposal) + (update_gate * self.hidden_state)

        return self.hidden_state

    def forward_sequence(self, sequence: Sequence[np.ndarray]) -> Sequence[np.ndarray]:

        self.reset_state()

        hidden_states = []

        for item in sequence:
            hidden_states.append(self.step(item))

        return hidden_states
