import pandas as pd
import pytest
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from canonical_sets import LUCID
from canonical_sets.data import Adult, DataSet
from canonical_sets.models import ClassifierPT, ClassifierTF


def test_inverse_design_error():

    with pytest.raises(
        ValueError,
        match="model must be a torch.nn.Module or \\(tf.\\)keras.Model.",
    ):
        LUCID(5, pd.DataFrame(), pd.DataFrame())


def test_inverse_design_pt():
    data = Adult()

    train_dataset = DataSet(data.train_data[:5000], data.train_labels[:5000])
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ClassifierPT(
        len(data.train_data.columns), len(data.train_labels.columns)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1):
        for x, y in train_dl:
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    example_data = data.train_data
    outputs = pd.DataFrame([[0, 1]], columns=["<=50K", ">50K"])

    lucid_index = LUCID(
        model,
        outputs,
        example_data,
        numb_of_epochs=5,
        lr=10,
        index=False,
        log_every_n=1,
    )

    lucid_pre_true = LUCID(
        model,
        outputs,
        example_data,
        numb_of_epochs=5,
        lr=10,
        one_hot_pre=True,
        log_every_n=1,
    )

    lucid_post_false = LUCID(
        model,
        outputs,
        example_data,
        numb_of_epochs=5,
        lr=10,
        extra_epoch=False,
        one_hot_post=False,
        log_every_n=1,
    )

    lucid = LUCID(
        model, outputs, example_data, numb_of_epochs=5, lr=10, log_every_n=1
    )

    lucid.process_results()
    lucid_index.process_results()

    assert len(lucid.results_processed.columns) == (
        len(lucid._output_columns)
        + len(lucid.categories)
        + len(lucid.numerical_cols)
    )

    assert all(
        lucid_index.results[["sample", "epoch"]]
        == lucid_index.results_processed[["sample", "epoch"]]
    )

    assert lucid_index.results.index.values.tolist() == list(range(600))

    assert all(
        lucid_pre_true.results.query("epoch == 1")[[">50K"]].values
        < lucid_pre_true.results.query("epoch == 5")[[">50K"]].values
    )

    assert all(
        lucid.results.query("epoch == 1")[[">50K"]].values
        < lucid.results.query("epoch == 5")[[">50K"]].values
    )

    for col in lucid.categories:
        col = col + "+"

        assert all(
            lucid_post_false.results.query("epoch != 1")
            .loc[:, lucid_post_false.results.columns.str.startswith(col)]
            .sum(1)
            .values
            != 1
        )

        assert all(
            lucid_pre_true.results.loc[
                :, lucid_pre_true.results.columns.str.startswith(col)
            ]
            .sum(1)
            .values
            == 1
        )

        assert all(
            lucid_pre_true.results.loc[
                :, lucid_pre_true.results.columns.str.startswith(col)
            ].sum()
            <= 590
        )

        assert all(
            lucid.results.loc[:, lucid.results.columns.str.startswith(col)]
            .sum(1)
            .values
            == 1
        )

        assert all(
            lucid.results.loc[
                :, lucid.results.columns.str.startswith(col)
            ].sum()
            <= 590
        )


def test_inverse_design_tf():
    data = Adult()

    model = ClassifierTF(len(data.train_labels.columns))

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        data.train_data[:5000].to_numpy(),
        data.train_labels[:5000].to_numpy(),
        epochs=1,
    )

    example_data = data.train_data
    outputs = pd.DataFrame([[0, 1]], columns=["<=50K", ">50K"])

    lucid_index = LUCID(
        model,
        outputs,
        example_data,
        numb_of_epochs=5,
        lr=10,
        index=False,
        log_every_n=1,
    )

    lucid_pre_true = LUCID(
        model,
        outputs,
        example_data,
        numb_of_epochs=5,
        lr=10,
        one_hot_pre=True,
        log_every_n=1,
    )

    lucid_post_false = LUCID(
        model,
        outputs,
        example_data,
        numb_of_epochs=5,
        lr=10,
        extra_epoch=False,
        one_hot_post=False,
        log_every_n=1,
    )

    lucid = LUCID(
        model, outputs, example_data, numb_of_epochs=5, lr=10, log_every_n=1
    )

    lucid.process_results()
    lucid_index.process_results()

    assert len(lucid.results_processed.columns) == (
        len(lucid._output_columns)
        + len(lucid.categories)
        + len(lucid.numerical_cols)
    )

    assert all(
        lucid_index.results[["sample", "epoch"]]
        == lucid_index.results_processed[["sample", "epoch"]]
    )

    assert lucid_index.results.index.values.tolist() == list(range(600))

    assert all(
        lucid_pre_true.results.query("epoch == 1")[[">50K"]].values
        < lucid_pre_true.results.query("epoch == 5")[[">50K"]].values
    )

    assert all(
        lucid.results.query("epoch == 1")[[">50K"]].values
        < lucid.results.query("epoch == 5")[[">50K"]].values
    )

    for col in lucid.categories:
        col = col + "+"

        assert all(
            lucid_post_false.results.query("epoch != 1")
            .loc[:, lucid_post_false.results.columns.str.startswith(col)]
            .sum(1)
            .values
            != 1
        )

        assert all(
            lucid_pre_true.results.loc[
                :, lucid_pre_true.results.columns.str.startswith(col)
            ]
            .sum(1)
            .values
            == 1
        )

        assert all(
            lucid_pre_true.results.loc[
                :, lucid_pre_true.results.columns.str.startswith(col)
            ].sum()
            <= 590
        )

        assert all(
            lucid.results.loc[:, lucid.results.columns.str.startswith(col)]
            .sum(1)
            .values
            == 1
        )

        assert all(
            lucid.results.loc[
                :, lucid.results.columns.str.startswith(col)
            ].sum()
            <= 590
        )
