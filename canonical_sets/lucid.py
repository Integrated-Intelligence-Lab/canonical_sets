"""LUCID."""

import warnings
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import default_rng
from sklearn.base import TransformerMixin
from tqdm import tqdm

from canonical_sets.utils import safe_isinstance


class LUCID:
    """Gradient-based inverse design to generate canonical sets.

    This class generates a canonical set via inverse design and
    attributes the ``pd.DataFrame``  to ``results``.

    Attributes
    ----------
    results : pd.DataFrame
        A dataframe with the canonical inputs.
    results_processed: pd.DataFrame
        A dataframe with the processed canonical inputs.

    Examples
    --------
    >>> model = tf.keras.Model()
    >>> outputs = pd.DataFrame([[0, 1]], columns=["No", "Yes"])
    >>> example_data = train_data
    >>> lucid = LUCID(model, outputs, example_data)
    >>> lucid.results
    """

    results: pd.DataFrame
    results_processed: pd.DataFrame

    def __init__(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        outputs: pd.DataFrame,
        example_data: pd.DataFrame,
        numb_of_samples: int = 100,
        numb_of_epochs: int = 200,
        lr: float = 0.1,
        low: float = -1,
        high: float = 1,
        seed: int = 1234,
        index: bool = True,
        extra_epoch: bool = True,
        one_hot_pre: bool = False,
        one_hot_post: bool = True,
        log_every_n: int = 0,
        prefix_sep: str = "+",
    ):
        """Initialize the inverse design.

        Parameters
        ----------
        model : torch.nn.Module or tf.keras.Model
            The trained model to use for inverse design.
        outputs : pd.DataFrame
            The outputs to use for inverse design. These are the
            targets/labels that have been used during training.
            For example, ``pd.DataFrame([[0, 1]], columns=["<=50K", ">50K"])``
            in the ``Adult`` data set.
        example_data : pd.DataFrame
            The example data to infer columns, dtypes, ...
            This is often (a part of) the training data itself,
            but can also be an artificial example.
        numb_of_samples : int
            The number of samples to generate. The default is 100.
        numb_of_epochs : int
            The number of epochs to train the model. The default is 200.
        lr : float
            The learning rate for the optimizer. The default is 0.1.
        low : float
            The lower bound for the random uniform distribution.
            The default is -1.
        high : float
            The upper bound for the random uniform distribution.
            The default is 1.
        seed : int
            The seed for the random number generator. The default is 1234.
        index : bool
            If True the sample and epoch numbers are used as indices in
            the results ``pd.DataFrame``. Otherwise they are just columns.
            The default is True.
        extra_epoch : bool
            If True an additional forward pass is run after the categorical
            features have been one-hot encoded (post-processed). The results
            are saved for the last sample as the ``numb_of_epochs`` + 1 epoch.
            If there are no categorical features the argument is ignored.
            The default is True.
        one_hot_pre : bool
            If True, the initial values for the categorical features are
            pre-processed to be one-hot. If there are no categorical
            features the argument is ignored. Note that the inverse
            design will start from this one-hot sample, hence the pre-
            process. If False, the inverse design will start from the
            randomly drawn initial vectors. The default is False.
        one_hot_post : bool
            If True, the values for the categorical features are
            post-processed to be one-hot. Note that the predictions during
            the inverse design are made with the original values of the
            categorical features and not with the post-processed values.
            To run an additional forward pass with the post-processed
            values check the ``extra_epoch`` argument. If there are no
            categorical features the argument is ignored. The default is True.
        log_every_n : int
            The number of epochs to log results. If 0, this argument is set
            equal to the ``numb_of_epochs`` argument which makes it a static
            analysis with only the start and end samples. The default is 0.
        prefix_sep : str
            The separator for the prefix of the column names. The one-hot
            encoded features are grouped via the prefix. To be safe, make
            sure that the prefix only appears as a prefix in the column
            names (i.e., avoid Categorical-category-name, and opt for
            Categorical+category-name instead). The default is "+".

        Raises
        ------
        ValueError
            If any columns are neither integer (one-hot encoded)
            or float (numerical).
        ValueError
            If the model is neither a torch.nn.Module or (tf.)keras.Model.
        """
        self.model = model
        self.outputs = outputs
        self.example_data = example_data
        self.numb_of_samples = numb_of_samples
        self.numb_of_epochs = numb_of_epochs
        self.lr = lr
        self.low = low
        self.high = high
        self.seed = seed
        self.index = index
        self.extra_epoch = extra_epoch
        self.one_hot_pre = one_hot_pre
        self.one_hot_post = one_hot_post
        self.log_every_n = log_every_n
        self.prefix_sep = prefix_sep

        if self.log_every_n == 0:
            self.log_every_n = self.numb_of_epochs

        self._processed = False

        # check whether all columns are either float or integer
        if len(self.example_data.columns) != len(
            self.example_data.select_dtypes(
                include=["float", "integer"]
            ).columns
        ):
            raise ValueError(
                (
                    "Some columns are neither integer (one-hot encoded) "
                    "or float (numerical)"
                )
            )

        # store column information
        self._input_columns = self.example_data.columns.to_list()
        self._numb_of_inputs = len(self._input_columns)

        self._output_columns = self.outputs.columns.to_list()

        self.numerical_cols = self.example_data.select_dtypes(
            include="float"
        ).columns.to_list()

        self.categories_cols = self.example_data.select_dtypes(
            include="integer"
        ).columns.to_list()

        self.categories = list(
            dict.fromkeys(
                [
                    item.split(self.prefix_sep)[0]
                    for item in self.categories_cols
                ]
            )
        )

        # generate random inputs
        self._rng = default_rng(seed=self.seed)
        self._inputs = self._rng.uniform(
            low=self.low,
            high=self.high,
            size=(self.numb_of_samples, self._numb_of_inputs),
        )

        # one-hot-encode categorical features (pre-processing, optional)
        if self.one_hot_pre and self.categories_cols:
            self._inputs = self._one_hot_encode(self._inputs)

        # initialize results
        self._columns = (
            ["sample", "epoch"] + self._output_columns + self._input_columns
        )
        self.results = pd.DataFrame(columns=self._columns)

        # run inverse design loop
        if safe_isinstance(self.model, "torch.nn.Module"):
            self._inverse_loop_pt()

        elif safe_isinstance(self.model, "keras.Model"):
            self._inverse_loop_tf()

        else:
            raise ValueError(
                "model must be a torch.nn.Module or (tf.)keras.Model."
            )

        # set types in results
        self.results["sample"] = self.results["sample"].astype(int)
        self.results["epoch"] = self.results["epoch"].astype(int)

        # save additional info
        self._min_epoch = 1
        self._max_epoch = self.numb_of_epochs - (
            self.numb_of_epochs % self.log_every_n
        )

        # set index
        if self.index:
            self.results.set_index(["sample", "epoch"], inplace=True)

        # one-hot-encode categorical features (post-processing, optional)
        if self.one_hot_post and self.categories_cols:
            self.results = self._one_hot_encode(self.results)

    def process_results(self, scaler: TransformerMixin = None) -> None:
        """Process the results by applying inverse scaler and one-hot
        encoding to categories.

        Parameters
        ----------
        scaler: sklearn.base.TransformerMixin, optional
            Any of the ``sklearn`` preprocessing modules.
            The default is None which means there is no transformation on
            numerical features.
        """
        if self._processed:
            warnings.warn("Results have already been processed before.")

        numbers = pd.DataFrame(index=self.results.index)
        dummies = pd.DataFrame(index=self.results.index)

        if self.numerical_cols and scaler:
            numbers = pd.DataFrame(
                scaler.inverse_transform(self.results[self.numerical_cols]),
                index=self.results.index,
                columns=self.numerical_cols,
            )

        elif self.numerical_cols:
            numbers = self.results[self.numerical_cols]

        if self.categories_cols:
            cols2collapse = {
                item.split(self.prefix_sep)[0]: (self.prefix_sep in item)
                for item in self.categories_cols
            }

            series_list = []
            for col, needs_to_collapse in cols2collapse.items():
                if needs_to_collapse:
                    undummified = (
                        self.results[self.categories_cols]
                        .filter(like=col)
                        .astype(float)
                        .idxmax(axis=1)
                        .apply(
                            lambda x: x.split(self.prefix_sep, maxsplit=1)[1]
                        )
                        .rename(col)
                    )
                    series_list.append(undummified)

                else:
                    series_list.append(self.results[col])
            dummies = pd.concat(series_list, axis=1)

        results_processed = numbers.join(dummies)

        self.results_processed = self.results[self._output_columns].join(
            results_processed
        )

        if self.index is False:
            self.results_processed = self.results[["sample", "epoch"]].join(
                self.results_processed
            )

        self._processed = True

    def plot(self, output: str) -> None:
        """Plot the outputs.

        Parameters
        ----------
        output: str
            The name of the output to plot.
        """
        if self.index:
            results = self.results

        else:
            results = self.results.set_index(["sample", "epoch"])

        plt.scatter(
            results.query(f"epoch == {self._min_epoch}")[
                output
            ].index.get_level_values("sample"),
            results.query(f"epoch == {self._min_epoch}")[output].values,
            color="blue",
        )

        plt.scatter(
            results.query(f"epoch == {self._max_epoch}")[
                output
            ].index.get_level_values("sample"),
            results.query(f"epoch == {self._max_epoch}")[output].values,
            color="red",
        )

        plt.scatter(
            results.query(f"epoch == {self._max_epoch + 1}")[
                output
            ].index.get_level_values("sample"),
            results.query(f"epoch == {self._max_epoch + 1}")[output].values,
            color="gray",
        )

        plt.title(f"{output}")

        plt.legend(
            [
                f"epoch {self._min_epoch}",
                f"epoch {self._max_epoch}",
                f"epoch {self._max_epoch + 1}",
            ]
        )

        plt.show()

    def hist(self, features: Union[str, List[str]]) -> None:
        """Plot the results for a given feature.

        Parameters
        ----------
        features: str or list of str
            The feature(s) to plot (either 1, 2, 3, 4, 6 or 8).

        Raises
        ------
        ValueError
            If the ``features`` are neither a string or a list of
            strings of size 2, 3, 4, 6 or 8.

        Note
        ----
        If the ``results`` are not yet processed, they will be
        with ``process_results``.
        """
        if not self._processed:
            self.process_results()

        if self.index:
            results = self.results_processed

        else:
            results = self.results_processed.set_index(["sample", "epoch"])

        if isinstance(features, str):
            plt.hist(
                results.query(f"epoch == {self._min_epoch}")[[features]],
                alpha=0.45,
                color="blue",
            )
            plt.hist(
                results.query(f"epoch == {self._max_epoch}")[[features]],
                alpha=0.45,
                color="red",
            )

            plt.title(f"{features}")

            plt.legend(
                [f"epoch {self._min_epoch}", f"epoch {self._max_epoch}"]
            )
            plt.show()

            return None

        if len(features) == 2:
            ncol = 2
            nrow = 1

        elif len(features) == 3:
            ncol = 3
            nrow = 1

        elif len(features) == 4:
            ncol = 2
            nrow = 2

        elif len(features) == 6:
            ncol = 3
            nrow = 2

        elif len(features) == 8:
            ncol = 4
            nrow = 2

        else:
            raise ValueError(
                "features must be either a string or list of "
                "size 2, 3, 4, 6 or 8."
            )

        fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
        for feature, ax in zip(features, axs.ravel()):
            ax.hist(
                results.query(f"epoch == {self._min_epoch}")[[feature]],
                alpha=0.45,
                color="blue",
            )

            ax.hist(
                results.query(f"epoch == {self._max_epoch}")[[feature]],
                alpha=0.45,
                color="red",
            )

            ax.set_title(f"{feature}")
            ax.legend([f"epoch {self._min_epoch}", f"epoch {self._max_epoch}"])

    def _one_hot_encode(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """One-hot encode the data.

        This method takes a dataframe or a numpy array and one-hot encodes
        the categorical features. The return type is the same as the input
        type.

        Parameters
        ----------
        data: pd.DataFrame or np.ndarray
            The data where the categorical features should be one-hot encoded.

        Returns
        -------
        results: pd.DataFrame or np.ndarray
            The data where the categorical features are one-hot encoded.
        """
        if isinstance(data, np.ndarray):
            is_numpy_array = True
            data = pd.DataFrame(data, columns=self._input_columns)

        else:
            is_numpy_array = False

        results = data.drop(self.numerical_cols + self.categories_cols, axis=1)

        _results = pd.DataFrame(index=results.index)
        categories = self.categories.copy()

        for col in self._input_columns:

            if col in self.numerical_cols:
                _results[col] = data[col]

            elif col.split(self.prefix_sep)[0] in categories:
                categories.remove(col.split(self.prefix_sep)[0])

                filtered = (
                    data[self.categories_cols]
                    .filter(like=col.split(self.prefix_sep)[0])
                    .astype(float)
                )
                df = pd.DataFrame(
                    np.transpose(
                        np.where(filtered.T == filtered.T.max(), 1, 0)
                    ),
                    index=data.index,
                    columns=filtered.columns,
                )

                _results = pd.concat([_results, df], axis=1)

        _results = _results[self._input_columns]

        results = results.join(_results)

        if is_numpy_array:
            return results.to_numpy()

        return results

    def _inverse_loop_pt(self) -> None:
        """Inverse design loop for PyTorch."""

        # transform output
        y = torch.tensor(self.outputs.to_numpy(), dtype=torch.float32)

        # inverse design loop
        for i, x in enumerate(tqdm(self._inputs)):
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            x.requires_grad_()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD([x], lr=self.lr)

            for epoch in range(self.numb_of_epochs):
                sample = x.detach().numpy().flatten()

                optimizer.zero_grad()

                outputs = self.model(x)
                loss = criterion(outputs, y)

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.log_every_n == 0 or epoch == 0:
                    pred = outputs.detach().numpy().flatten()

                    row = np.concatenate(
                        [
                            np.array([(i + 1), (epoch + 1)]),
                            pred,
                            sample,
                        ]
                    )

                    self.results = pd.concat(
                        [
                            self.results,
                            pd.DataFrame([row], columns=self._columns),
                        ],
                        ignore_index=True,
                    )

            if self.categories_cols and self.extra_epoch:
                sample = self._one_hot_encode(np.expand_dims(sample, 0))

                output = self.model(torch.Tensor(sample))
                pred = output.detach().numpy().flatten()

                row = np.concatenate(
                    [np.array([(i + 1), (epoch + 2)]), pred, sample.squeeze()]
                )

                self.results = pd.concat(
                    [
                        self.results,
                        pd.DataFrame([row], columns=self._columns),
                    ],
                    ignore_index=True,
                )

    def _inverse_loop_tf(self) -> None:
        """Inverse design loop for Tensorflow."""
        # transform output
        y = self.outputs.to_numpy()

        # inverse design loop
        for i, x in enumerate(tqdm(self._inputs)):
            x = tf.Variable(np.expand_dims(x, axis=0))

            criterion = tf.keras.losses.BinaryCrossentropy()
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

            for epoch in range(self.numb_of_epochs):
                sample = x.numpy()

                with tf.GradientTape() as tape:
                    tape.watch(x)
                    outputs = self.model(x, training=False)
                    loss = criterion(y, outputs)

                grads = tape.gradient(loss, x)
                optimizer.apply_gradients(zip([grads], [x]))

                if (epoch + 1) % self.log_every_n == 0 or epoch == 0:
                    pred = outputs.numpy()

                    row = np.concatenate(
                        [
                            np.array([[(i + 1), (epoch + 1)]]),
                            pred,
                            sample,
                        ],
                        axis=1,
                    )

                    self.results = pd.concat(
                        [
                            self.results,
                            pd.DataFrame(
                                [row.squeeze()], columns=self._columns
                            ),
                        ],
                        ignore_index=True,
                    )

            if self.categories_cols and self.extra_epoch:
                sample = self._one_hot_encode(sample)

                output = self.model(sample, training=False)
                pred = output.numpy()

                self.sample = sample
                self.output = output
                self.pred = pred

                row = np.concatenate(
                    [
                        np.array([(i + 1), (epoch + 2)]),
                        pred.squeeze(),
                        sample.squeeze(),
                    ]
                )

                self.results = pd.concat(
                    [
                        self.results,
                        pd.DataFrame([row], columns=self._columns),
                    ],
                    ignore_index=True,
                )
