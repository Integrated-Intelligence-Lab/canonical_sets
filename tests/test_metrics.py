import numpy as np

from canonical_sets.group import Metrics


def test_metrics():
    preds = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    targets = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    metrics = Metrics(preds, targets)

    assert metrics.metrics == {"acc": 50.0, "pr": 50.0, "tpr": 50.0}
