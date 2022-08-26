"""Classifiers."""

import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F


class ClassifierPT(nn.Module):
    """Classifier for PyTorch.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = F.softmax(self.linear(x), dim=1)
        return outputs


class ClassifierTF(tf.keras.Model):
    """Classifier for Keras.

    Parameters
    ----------
    output_dim : int
        Output dimension.
    """

    def __init__(self, output_dim: int):
        super().__init__()

        self.linear = tf.keras.layers.Dense(
            output_dim, activation=tf.nn.softmax
        )

    def call(self, x):
        outputs = self.linear(x)
        return outputs
