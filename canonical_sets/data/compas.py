"""Compas Data Set - ProPublica."""

from typing import Dict, List, Optional

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from canonical_sets.data.base import BaseData


class Compas(BaseData):
    """Compas Data Set - ProPublica.

    This class downloads and preprocesses the Compas dataset as
    a `pd.DataFrame`.

    Attributes
    ----------
    train_data : pd.DataFrame
        The training data.
    test_data : pd.DataFrame
        The testing data.
    train_labels : pd.DataFrame
        The training labels.
    test_labels : pd.DataFrame
        The testing labels.
    val_data : pd.DataFrame
        The validation data.
    val_labels : pd.DataFrame
        The validation labels.
    numerical_cols : List[str]
        The numerical columns.
    categorical_cols : List[str]
        The categorical columns.

    Example
    -------
    >>> compas = Compas()
    """

    train_data: pd.DataFrame
    val_data: Optional[pd.DataFrame]
    test_data: Optional[pd.DataFrame]
    train_labels: Optional[pd.DataFrame]
    val_labels: Optional[pd.DataFrame]
    test_labels: Optional[pd.DataFrame]
    numerical_cols: Optional[List[str]]
    categorical_cols: Optional[List[str]]

    def __init__(
        self,
        path: Optional[str] = None,
        download_path: Optional[str] = None,
        features: Optional[List[str]] = None,
        groups: Optional[Dict[str, Dict[str, str]]] = None,
        scaler: TransformerMixin = MinMaxScaler(feature_range=(-1, 1)),
        prefix_sep: str = "+",
        val_prop: float = 0.2,
        test_prop: float = 0.2,
        preprocess: bool = True,
        seed: int = 1234,
    ):
        """Initialize the data.

        Parameters
        ----------
        path : Optional[str]
            The path to the data if it is already downloaded.
        download_path : Optional[str]
            The path to save the data to (needs to end in .csv).
            The default is ``None``.
        features: List[str], optional
            The features to use. The default is ``None``.
        groups: Dict[str, Dict[str, str]], optional
            The groups to use. The default is ``None``.
        scaler : sklearn.base.TransformerMixin
            Any of the ``sklearn`` preprocessing modules.
            The default is ``sklearn.preprocessing.MinMaxScaler``.
        prefix_sep : str
            The prefix separator to split the categorical feature and category
            when one-hot encoding. For example, Color = [Red, Green] ->
            Color+Red and Color+Green. The default is ``+``.
        val_prop: float
            The proportion of the training data (minus the testing data)
            to use for validation. The default is 0.2.
        test_prop: float
            The proportion of the training data to use for testing.
             The default is 0.2.
        preprocess: bool
            Whether to preprocess the data. The default is ``True``.
        seed: int
            The seed for the random state. The default is 1234.
        """
        super().__init__(
            features,
            groups,
            scaler,
            prefix_sep,
            val_prop,
            test_prop,
            preprocess,
            seed,
        )

        if self.features:
            self.features = features
        else:
            self.features = [
                "c_charge_degree",
                "race",
                "age_cat",
                "sex",
                "priors_count",
            ]

        if path:
            data = pd.read_csv(path)

        else:
            url = (
                "https://raw.githubusercontent.com/propublica/"
                "compas-analysis/master/compas-scores-two-years.csv"
            )

            data = pd.read_csv(url)

            if download_path:
                data.to_csv(download_path, index=False)

        if preprocess:
            self._preprocess(data)

        else:
            self.train_data = data

            self.test_data = None
            self.val_data = None
            self.train_labels = None
            self.val_labels = None
            self.test_labels = None
            self.numerical_cols = None
            self.categorical_cols = None

    def _preprocess(self, data: pd.DataFrame) -> None:
        """Preprocess the data.

        Parameters
        ----------
        data: pd.DataFrame
            The data.

        Returns
        -------
        None.
        """

        # cleaning data like in the original ProPublica paper
        df = (
            data.loc[
                (data["days_b_screening_arrest"] <= 30)
                & (data["days_b_screening_arrest"] >= -30),
                :,
            ]
            .loc[data["is_recid"] != -1, :]
            .loc[data["c_charge_degree"] != "O", :]
            .loc[data["score_text"] != "N/A", :]
        )

        df.reset_index(drop=True, inplace=True)

        # get labels
        labels = pd.get_dummies(df["two_year_recid"])
        df.drop("two_year_recid", inplace=True, axis=1)

        # drop columns
        df = df[self.features]

        # create groups
        if self.groups is not None:
            df = self._create_groups(df, self.groups)

        # split data
        if self.val_prop > 0 and self.test_prop > 0:
            (
                x_train,
                x_test,
                y_train,
                y_test,
            ) = self._split_data(df, labels, self.test_prop)

            (
                x_train,
                x_val,
                y_train,
                y_val,
            ) = self._split_data(x_train, y_train, self.val_prop)

            data = pd.concat([x_train, x_val, x_test])

        elif self.val_prop > 0:
            (
                x_train,
                x_val,
                y_train,
                y_val,
            ) = self._split_data(df, labels, self.val_prop)

            data = pd.concat([x_train, x_val])

        elif self.test_prop > 0:
            (
                x_train,
                x_test,
                y_train,
                y_test,
            ) = self._split_data(df, labels, self.test_prop)

            data = pd.concat([x_train, x_test])

        # one-hot encode categorical columns
        dummies = self._one_hot_encode(data)

        # scale numerical columns
        scaled_train_data = self._scale_numeric(x_train, fit_scaler=True)

        if self.val_prop > 0:
            scaled_val_data = self._scale_numeric(x_val)

        if self.test_prop > 0:
            scaled_test_data = self._scale_numeric(x_test)

        # merge the pre-processed data and attribute to self
        self.train_data = scaled_train_data.join(dummies[: len(x_train)])
        self.train_labels = y_train

        if self.val_prop > 0 and self.test_prop > 0:
            self.val_data = scaled_val_data.join(
                dummies[len(x_train) : (len(x_train) + len(x_val))]
            )
            self.test_data = scaled_test_data.join(
                dummies[(len(x_train) + len(x_val)) :]
            )

            self.val_labels = y_val
            self.test_labels = y_test

        elif self.val_prop > 0:
            self.val_data = scaled_val_data.join(dummies[len(x_train) :])

            self.val_labels = y_val

        elif self.test_prop > 0:
            self.test_data = scaled_test_data.join(dummies[len(x_train) :])

            self.test_labels = y_test
