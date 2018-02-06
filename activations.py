#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Müller / mmueller@cl.uzh.ch

import numpy as np


def sigmoid(inputs: np.ndarray) -> np.ndarray:
    """
    Squashing non-linearity sigmoid.
    """
    return 1 / (1 + np.exp(- inputs))

def tanh(inputs: np.ndarray) -> np.ndarray:
  """
  Hyperbolic tangent non-linearity.
  """
  return np.tanh(inputs)

def linear(inputs):

    return inputs
