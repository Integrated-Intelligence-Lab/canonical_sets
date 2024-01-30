"""Sampler."""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import SpanInfo


class _Sampler(DataSampler):
    """Sampler wrapping ``DataSampler``.

    This class is based on the `DataSampler` class from the `ctgan`
    package. It has been modified to fix several bugs (see PRs on
    the `ctgan` GitHub page) and to allow for the extension of the
    conditional vector. Note that a part of the code and
    comments is identical to the original `DataSampler` class.
    """

    def __init__(
        self,
        data: np.ndarray,
        output_info: List[List[SpanInfo]],
        log_frequency: bool,
        conditions: Optional[np.ndarray] = None,
    ):
        """Initialize the sampler.

        Parameters
        ----------
        data : numpy.ndarray
            The data to be sampled.
        output_info : ctgan.data_transformer.SpanInfo
            The output information from the DataTransformer.
        log_frequency : bool
            Whether to use the log frequency.
        conditions : numpy.ndarray, optional
            The conditions to be sampled.
        """
        self._data = torch.from_numpy(data.astype("float32"))

        self._conditions = (
            torch.from_numpy(conditions.astype("float32"))
            if conditions is not None
            else None
        )

        def is_discrete_column(column_info):
            return (
                len(column_info) == 1
                and column_info[0].activation_fn == "softmax"
            )

        n_discrete_columns = sum(
            [
                1
                for column_info in output_info
                if is_discrete_column(column_info)
            ]
        )

        self._discrete_column_matrix_st = np.zeros(
            n_discrete_columns, dtype="int32"
        )

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix to efficiently sample conditional vector
        max_category = max(
            [
                column_info[0].dim
                for column_info in output_info
                if is_discrete_column(column_info)
            ],
            default=0,
        )

        self._discrete_column_cond_st = np.zeros(
            n_discrete_columns, dtype="int32"
        )
        self._discrete_column_n_category = np.zeros(
            n_discrete_columns, dtype="int32"
        )
        self._discrete_column_category_prob = np.zeros(
            (n_discrete_columns, max_category)
        )
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum(
            [
                column_info[0].dim
                for column_info in output_info
                if is_discrete_column(column_info)
            ]
        )

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[
                    current_id, : span_info.dim
                ] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                self._discrete_column_matrix_st[current_id] = st

                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    def sample_original_condvec(
        self, batch: int
    ) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
        """Generate the conditional vector using the original frequency.

        Parameters
        ----------
        batch : int
            The batch size.

        Returns
        -------
        np.ndarray, torch.Tensor, optional
            The conditional vector and (optionally) extra conditions.
        """
        #Adjustment: Otherwise not possible to return None if no categorical variables or conditions!!!
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype="float32")

        if self._conditions is not None:
            ids = []

        for i in range(batch):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.randint(0, self._n_discrete_columns)
            matrix_st = self._discrete_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

            if self._conditions is not None:
                ids.append(row_idx)

        if self._conditions is not None:
            return cond, self._conditions[ids]
        else:
            return cond

    def sample_data(
        self, n: int, col: Optional[np.ndarray] = None, opt: Optional[np.ndarray] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample data from original training data for the conditional vector.

        Parameters
        ----------
        n : int
            The number of samples.
        col : np.ndarray, optional
            The column to be sampled.
        opt : np.ndarray, optional
            The category to be sampled.

        Returns
        -------
        torch.Tensor
            The conditional vector and (optionally) extra conditions.
        """

        #adjustment: col and opt are no integers! They are np.ndarrays!!! (they can be integers but not in this workflow)
        if col is None and opt is None:
            idx = np.random.randint(len(self._data), size=n)

            if self._conditions is None:
                return self._data[idx, :]

            else:
                return self._data[idx, :], self._conditions[idx, :]

        else:
            idx_list = []
            for c, o in zip(col, opt):  # type: ignore
                idx_list.append(np.random.choice(self._rid_by_cat_cols[c][o]))

            if self._conditions is None:
                return self._data[idx_list, :]

            else:
                return self._data[idx_list, :], self._conditions[idx_list, :]

    def generate_cond_from_condition_column_info(
        self, condition_infos: List[Dict[str, int]], batch: int
    ) -> np.ndarray:
        """Generate the condition vector.

        Parameters
        ----------
        condition_infos : List[Dict[str, int]]
            The condition column information.
        batch : int
            The batch size.

        Returns
        -------
        numpy.ndarray
            The condition vector.
        """
        vec = np.zeros((batch, self._n_categories), dtype="float32")

        for condition_info in condition_infos:
            id_ = self._discrete_column_cond_st[
                condition_info["discrete_column_id"]
            ]
            id_ += condition_info["value_id"]
            vec[:, id_] = 1

        return vec
