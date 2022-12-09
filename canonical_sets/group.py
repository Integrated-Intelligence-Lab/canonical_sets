"""Output-based group metrics."""
import numpy as np
from sklearn.metrics import confusion_matrix


class Metrics:
    """Class to compute output-based metrics on binary classification tasks.

    This class uses predictions and targets to compute the accuracy,
    positivity rate, and true positive rate on binary classification tasks.
    The disparities between positivity rates can be used to assess
    Demographic Parity, and the disparities between true positive rates can
    be used to assess Equality of Opportunity. See
    https://dl.acm.org/doi/abs/10.1145/3468507.3468511
    for more information.

    Attributes
    ----------
    metrics : Dict[str, float]
        Dict with the accuracy, positivity rate, and true positive rate on a
        binary classification task.

    Examples
    --------
    >>> import numpy as np
    >>> from canonical_sets.group import Metrics
    >>> preds = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    >>> targets = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    >>> metrics = Metrics(preds, targets)
    >>> metrics.metrics
    """

    def __init__(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
    ):
        """Initialize FairMetrics class.

        Parameters
        ----------
        preds : np.ndarray
            A 1-dimensional array with the predictions of the model.
            These should be the estimated targets, i.e. [0, 1] for
            binary classification. Not the estimated probabilities,
            i.e. [0.1, 0.9].
        targets : np.ndarray
            A 1-dimensional array with the ground truth targets.
            These should be the true targets, i.e. [0, 1] for binary
            classification.
        """
        cm = confusion_matrix(targets, preds)

        tn, fp, fn, tp = cm.ravel()

        self.metrics = {
            "acc": np.around((tp + tn) / (tp + tn + fp + fn) * 100, 1),
            "pr": np.around((tp + fp) / (tp + tn + fp + fn) * 100, 1),
            "tpr": np.around(tp / (tp + fn) * 100, 1),
        }
