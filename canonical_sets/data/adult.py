"""Adult Data Set - UCI Machine Learning Repository."""

from typing import Dict, List, Optional

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from canonical_sets.data.base import BaseData

# names of the columns in the downloaded csv file
NAMES = [
    "Age",
    "Workclass",
    "fnlwgt",
    "Education",
    "Education-Num",
    "Martial Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Capital Gain",
    "Capital Loss",
    "Hours per week",
    "Country",
    "Target",
]


class Adult(BaseData):
    """Adult Data Set - UCI Machine Learning Repository.

    This class downloads and preprocesses the Adult dataset as
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
    >>> adult = Adult()
    """

    train_data: pd.DataFrame
    val_data: Optional[pd.DataFrame]
    test_data: pd.DataFrame
    train_labels: Optional[pd.DataFrame]
    val_labels: Optional[pd.DataFrame]
    test_labels: Optional[pd.DataFrame]
    numerical_cols: Optional[List[str]]
    categorical_cols: Optional[List[str]]

    def __init__(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        download_train_path: Optional[str] = None,
        download_test_path: Optional[str] = None,
        features: Optional[List[str]] = None,
        groups: Optional[Dict[str, Dict[str, str]]] = None,
        scaler: TransformerMixin = MinMaxScaler(feature_range=(-1, 1)),
        prefix_sep: str = "+",
        val_prop: float = 0.2,
        preprocess: bool = True,
        seed: int = 1234,
    ):
        """Initialize the data.

        Parameters
        ----------
        train_path : str, optional
            The path to the training data if it is already downloaded.
        test_path : str, optional
            The path to the testing data if it is already downloaded.
        download_train_path : str, optional
            The path to save the training data to (needs to end in .csv).
            The default is ``None``.
        download_test_path : str, optional
            The path to save the testing data to (needs to end in .csv).
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
            The proportion of the training data to use for validation.
            The default is 0.2.
        preprocess: bool
            Whether to preprocess the data. The default is ``True``.
        seed: int
            The seed for the random state. The default is 1234.
        """
        super().__init__(
            features, groups, scaler, prefix_sep, val_prop, 0, preprocess, seed
        )

        if self.features:
            self.features = features
        else:
            self.features = [
                "Age",
                "Workclass",
                "fnlwgt",
                "Education",
                "Education-Num",
                "Martial Status",
                "Occupation",
                "Relationship",
                "Race",
                "Sex",
                "Capital Gain",
                "Capital Loss",
                "Hours per week",
                "Country",
            ]

        if train_path and test_path:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

        else:
            train_url = (
                "https://archive.ics.uci.edu/ml/"
                "machine-learning-databases/adult/adult.data"
            )
            test_url = (
                "https://archive.ics.uci.edu/ml/"
                "machine-learning-databases/adult/adult.test"
            )

            train_data = pd.read_csv(
                train_url,
                names=NAMES,
                sep=r"\s*,\s*",
                engine="python",
                na_values="?",
                header=None,
            )

            test_data = pd.read_csv(
                test_url,
                names=NAMES,
                sep=r"\s*,\s*",
                engine="python",
                na_values="?",
                header=None,
                skiprows=1,
            )

            if download_train_path:
                train_data.to_csv(download_train_path, index=False)

            if download_test_path:
                test_data.to_csv(download_test_path, index=False)

        if preprocess:
            self._preprocess(train_data, test_data)

        else:
            self.train_data = train_data
            self.test_data = test_data

            self.val_data = None
            self.train_labels = None
            self.val_labels = None
            self.test_labels = None
            self.numerical_cols = None
            self.categorical_cols = None

    def _preprocess(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> None:
        """Preprocess the data.

        Parameters
        ----------
        train_data: pd.DataFrame
            The training data.
        test_data: pd.DataFrame
            The testing data.

        Returns
        -------
        None.
        """

        # drop NA
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)

        train_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        df = pd.concat([train_data, test_data])

        # get labels, drop target (erase unnecessary "." in test data)
        #Adjustment: No regex=True in 'replace()' !!!
        df["Target"] = df["Target"].str.replace(r".", "")

        labels = pd.get_dummies(df["Target"])
        df.drop("Target", inplace=True, axis=1)

        # drop columns
        df = df[self.features]

        # create groups
        # if self.groups is None:
        #     others = list(df.Country.unique())
        #     others.remove("United-States")
        #     self.groups = {"Country": dict.fromkeys(others, "Others")}

        if self.groups is not None:
            df = self._create_groups(df, self.groups)

        # split data
        x_train = df.iloc[: len(train_data)]
        y_train = labels.iloc[: len(train_data)]

        x_test = df.iloc[len(train_data) :]
        y_test = labels.iloc[len(train_data) :]

        if self.val_prop > 0:
            (
                x_train,
                x_val,
                y_train,
                y_val,
            ) = self._split_data(x_train, y_train, self.val_prop)

            data = pd.concat([x_train, x_val, x_test])

        else:
            data = pd.concat([x_train, x_test])

        # one-hot encode categorical columns
        dummies = self._one_hot_encode(data)

        # scale numerical columns
        scaled_train_data = self._scale_numeric(x_train, fit_scaler=True)
        scaled_test_data = self._scale_numeric(x_test)

        if self.val_prop > 0:
            scaled_val_data = self._scale_numeric(x_val)

        # merge the pre-processed data and attribute to self
        self.train_data = scaled_train_data.join(dummies[: len(x_train)])
        self.train_labels = y_train

        self.test_labels = y_test

        if self.val_prop > 0:
            self.val_data = scaled_val_data.join(
                dummies[len(x_train) : (len(x_train) + len(x_val))]
            )
            self.test_data = scaled_test_data.join(
                dummies[(len(x_train) + len(x_val)) :]
            )

            self.val_labels = y_val

        else:
            self.test_data = scaled_test_data.join(dummies[len(x_train) :])
