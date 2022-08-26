import tensorflow as tf
import torch.nn as nn

from canonical_sets.utils import safe_isinstance


def test_safe_isinstance():
    model_pt = nn.Module()
    result_pt = safe_isinstance(model_pt, "torch.nn.Module")

    model_tf = tf.keras.Model()
    result_tf = safe_isinstance(model_tf, "keras.Model")

    result_tf_false = safe_isinstance(model_tf, "tf.keras.Model")

    result_no_str = safe_isinstance(int, int)

    assert result_pt
    assert result_tf
    assert result_tf_false is False
    assert result_no_str is False
