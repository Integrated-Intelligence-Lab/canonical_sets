import numpy as np
import pandas as pd
from torch import is_tensor

from canonical_sets.data import DataSet


def test_dataset():
    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "continuous": rng.random(100),
        }
    )
    labels = pd.DataFrame({"targets": rng.integers(2, size=100)})

    data = DataSet(df, labels)

    sample, label = next(iter(data))

    assert is_tensor(sample)
    assert is_tensor(label)
    assert sample.shape.numel() == 1
    assert label.shape.numel() == 1
