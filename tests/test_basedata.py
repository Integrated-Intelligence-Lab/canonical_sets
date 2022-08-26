import numpy as np
import pandas as pd
import pytest

from canonical_sets.data.base import BaseData


def test_prop_fail():

    with pytest.raises(
        ValueError, match="Proportions must be between \\[0, 1\\)."
    ):
        BaseData(val_prop=-0.3)

    with pytest.raises(
        ValueError, match="Proportions must be between \\[0, 1\\)."
    ):
        BaseData(val_prop=1.3)

    with pytest.raises(
        ValueError, match="Proportions must be between \\[0, 1\\)."
    ):
        BaseData(test_prop=-0.3)

    with pytest.raises(
        ValueError, match="Proportions must be between \\[0, 1\\)."
    ):
        BaseData(test_prop=-1.3)


def test_create_groups():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "continuous": rng.random(100),
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )

    groups = {"discrete": {"b": "others", "c": "others"}}

    results = data._create_groups(df, groups)

    assert results.shape == (100, 2)
    assert all(i in results["discrete"].unique() for i in ["a", "others"])
    assert all(i not in results["discrete"].unique() for i in ["b", "c"])


def test_split_data():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "continuous": rng.random(100),
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )
    labels = pd.DataFrame({"targets": rng.integers(2, size=100)})

    x_data, y_data, x_labels, y_labels = data._split_data(df, labels, 0.5)

    assert x_data.shape == y_data.shape
    assert x_labels.shape == y_labels.shape


def test_one_hot_encode():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "continuous": rng.random(100),
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )

    results = data._one_hot_encode(df)

    assert results.columns.to_list() == [
        "discrete+a",
        "discrete+b",
        "discrete+c",
    ]
    assert all(results.sum(axis=1) == 1)


def test_one_hot_encode_no_cats():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "continuous": rng.random(100),
        }
    )

    results = data._one_hot_encode(df)

    assert results.empty


def test_inverse_ohe():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "continuous": rng.random(100),
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )

    results = data._one_hot_encode(df)

    results_inv = data._inverse_ohe(results)

    assert all(df[["discrete"]] == results_inv)
    assert results_inv.shape == (100, 1)
    assert pd.api.types.is_object_dtype(results_inv["discrete"])
    assert all(i in results_inv["discrete"].unique() for i in ["a", "b", "c"])


def test_inverse_ohe_no_cats():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "continuous": rng.random(100),
        }
    )

    results = data._inverse_ohe(df)

    assert results.empty


def test_scale_numeric():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df_train = pd.DataFrame(
        {
            "continuous": rng.random(100),
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )
    df_test = pd.DataFrame(
        {
            "continuous": rng.random(100),
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )

    results_train = data._scale_numeric(df_train, True)
    results_test = data._scale_numeric(df_test)

    assert results_train.shape == (100, 1)
    assert results_test.shape == (100, 1)

    assert np.allclose(results_train.min(), -1)
    assert np.allclose(results_train.max(), 1)


def test_scale_numeric_no_num():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )

    results = data._scale_numeric(df, True)

    assert results.empty


def test_inverse_scale_numeric():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df_train = pd.DataFrame(
        {
            "continuous": rng.random(100),
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )
    df_test = pd.DataFrame(
        {
            "continuous": rng.random(100),
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )

    results_train = data._scale_numeric(df_train, True)
    results_test = data._scale_numeric(df_test)

    results_train_inv = data._inverse_scale_numeric(results_train)
    results_test_inv = data._inverse_scale_numeric(results_test)

    assert (
        results_train_inv.eq(df_train[["continuous"]]).sum().continuous >= 80
    )
    assert results_test_inv.eq(df_test[["continuous"]]).sum().continuous >= 80


def test_inverse_scale_numeric_no_num():
    data = BaseData()

    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "discrete": rng.choice(["a", "b", "c"], 100),
        }
    )

    results = data._inverse_scale_numeric(df)

    assert results.empty
