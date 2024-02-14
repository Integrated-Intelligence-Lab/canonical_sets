"""Base class for data sets."""

from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class BaseData:
    """Base class for data sets.

    This is a base class from which all data sets inherit.

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
    """

    train_data: pd.DataFrame
    val_data: Optional[pd.DataFrame]
    test_data: Optional[pd.DataFrame]
    train_labels: pd.DataFrame
    val_labels: Optional[pd.DataFrame]
    test_labels: Optional[pd.DataFrame]
    numerical_cols: Optional[List[str]]
    categorical_cols: Optional[List[str]]

    def __init__(
        self,
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
        features: List[str], optional
            The features to use. The default is ``None``.
        groups: Dict[str, Dict[str, str]], optional
            The groups to use. The default is ``None``.
        scaler : sklearn.base.TransformerMixin
            Any of the ``sklearn`` preprocessing modules for the numerical
            features. The default is ``sklearn.preprocessing.MinMaxScaler``.
        prefix_sep : str
            The prefix separator to split the categorical feature and category
            when one-hot encoding. For example, Color = [Red, Green] ->
            Color+Red and Color+Green. The default is ``+``.
        val_prop: float
            The proportion of the training data to use for validation.
            The default is 0.2.
        test_prop: float
            The proportion of the training data to use for testing.
             The default is 0.2.
        preprocess: bool
            Whether to preprocess the data. The default is ``True``.
        seed: int
            The seed for the random state. The default is 1234.

        Raises
        ------
        ValueError
            Proportions must be between [0, 1].
        """
        if val_prop >= 1 or test_prop >= 1 or val_prop < 0 or test_prop < 0:
            raise ValueError("Proportions must be between [0, 1].")

        self.features = features
        self.groups = groups
        self.scaler = scaler
        self.prefix_sep = prefix_sep
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.preprocess = preprocess
        self.seed = seed

        self.train_data = pd.DataFrame()
        self.val_data = None
        self.test_data = None

        self.train_labels = pd.DataFrame()
        self.val_labels = None
        self.test_labels = None

        self.numerical_cols = None
        self.categorical_cols = None

    @classmethod
    def load(cls, path):
        """Load the data.

        Parameters
        ----------
        path: str
            The path to load the data from (needs to end in .pkl).
        """
        return joblib.load(path)

    def save(self, path: str) -> None:
        """Save the object.

        Parameters
        ----------
        path: str
            The path to save the object (needs to end in .pkl).
        """
        joblib.dump(self, path)

    def inverse_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse preprocess the data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to inverse preprocess.

        Returns
        -------
        pd.DataFrame
            The inverse preprocessed data.
        """
        numbers = self._inverse_scale_numeric(data)
        dummies = self._inverse_ohe(data)

        return numbers.join(dummies)

    def _create_groups(
        self, data: pd.DataFrame, groups: Dict[str, Dict[str, str]]
    ) -> pd.DataFrame:
        """Create groups for the data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to group.
        groups: Dict[str, Dict[str, str]]
            The groups to use. The default is ``None``.

        Returns
        -------
        df: pd.DataFrame
            The grouped data.
        """
        df = data.replace(groups)

        return df

    def _split_data(
        self,
        data: pd.DataFrame,
        labels: pd.DataFrame,
        prop: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to split.
        labels: pd.DataFrame
            The labels to split.
        prop: int
            The proportion of the data to use. The default is 0.2.

        Returns
        -------
        x_data: pd.DataFrame
            The x data.
        y_data: pd.DataFrame
            The y data.
        x_labels: pd.DataFrame
            The x labels.
        y_labels: pd.DataFrame
            The y labels.
        """
        x_data, y_data, x_labels, y_labels = train_test_split(
            data, labels, test_size=prop, random_state=self.seed
        )

        x_data.reset_index(drop=True, inplace=True)
        y_data.reset_index(drop=True, inplace=True)

        x_labels.reset_index(drop=True, inplace=True)
        y_labels.reset_index(drop=True, inplace=True)

        return x_data, y_data, x_labels, y_labels

    def _one_hot_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to one-hot encode.

        Returns
        -------
        data: pd.DataFrame
            The one-hot encoded data.
        """
        categories = data.select_dtypes(exclude="number")

        if categories.empty:
            return pd.DataFrame(index=data.index)

        self.categorical_cols = categories.columns.to_list()
        dummies = pd.get_dummies(
            categories,
            prefix=self.categorical_cols,
            prefix_sep=self.prefix_sep,
        )

        return dummies

    def _scale_numeric(
        self, data: pd.DataFrame, fit_scaler: bool = False
    ) -> pd.DataFrame:
        """Scale the numeric data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to scale.
        fit_scaler: bool
            Whether to fit the scaler. The default is ``False``.

        Returns
        -------
        data: pd.DataFrame
            The scaled data.
        """
        numerical_data = data.select_dtypes(include="number")

        if numerical_data.empty:
            return pd.DataFrame(index=data.index)

        if fit_scaler:
            self.scaler.fit(numerical_data)
            self.numerical_cols = numerical_data.columns.to_list()

        scaled_data = pd.DataFrame(
            self.scaler.transform(numerical_data),
            columns=self.numerical_cols,
        )

        return scaled_data

    def _inverse_ohe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse one-hot encode the data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to inverse one-hot encode.

        Returns
        -------
        undummified_df: pd.DataFrame
            The inverse one-hot encoded data.
        """
        # Adjustment: include='uint8' -> exclude="number" #
        df = data.select_dtypes(exclude="number")

        if df.empty or self.categorical_cols is None:
            return pd.DataFrame(index=data.index)

        cols2collapse = {
            item.split(self.prefix_sep)[0]: (self.prefix_sep in item)
            for item in df.columns
        }

        series_list = []
        for col, needs_to_collapse in cols2collapse.items():
            if needs_to_collapse & (col in self.categorical_cols):
                undummified = (
                    df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(self.prefix_sep, maxsplit=1)[1])
                    .rename(col)
                )
                series_list.append(undummified)
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)

        return undummified_df

    def _inverse_scale_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse scale the numeric data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to inverse scale.

        Returns
        -------
        pd.DataFrame
            The inverse scaled data.
        """
        df = data.select_dtypes(include="float64")

        if df.empty:
            return pd.DataFrame(index=data.index)

        return pd.DataFrame(
            self.scaler.inverse_transform(df), columns=df.columns
        )


class DataSet(Dataset):
    """The dataset class for the PyTorch dataloader."""

    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        """Initialize the dataset.
        Parameters
        ----------
        data: pd.DataFrame
            The data.
        labels: pd.DataFrame
            The labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.Tensor(self.data.iloc[idx])
        y = torch.Tensor(self.labels.iloc[idx])

        return x, y
