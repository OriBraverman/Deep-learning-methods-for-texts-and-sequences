"""
'experiment.py' file does as follows:
-------------------------------------
- Train and test a Recurrent Neural Network (RNN) acceptor that can preform
    binary classification on the input sequences.
- The RNN is specifically LSTM (Long Short-Term Memory) network followed by a
    MLP (Multi-Layer Perceptron) classifier with a single hidden layer.
- Formally, for an input sequence of n vectors x_1, x_2, ..., x_n,
    the output of the network is MLP(LSTM(x_1, x_2, ..., x_n)).
    where LSTM(x_1, x_2, ..., x_n) is mapping the sequence into a vector in R^d_1
    and MLP is used to classify this d_1-dimensional vector.


The language is defined as follows:
Positive examples:
------------------
[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+

Negative examples:
------------------
[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+
"""

# Importing necessary libraries
import os
import argparse
import torch

import l