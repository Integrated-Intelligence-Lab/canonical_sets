from sklearn.preprocessing import StandardScaler

from canonical_sets.data import Compas


def test_compas_no_preprocess():
    data = Compas(preprocess=False)

    assert data.train_data.shape == (7214, 53)


def test_compas():
    data = Compas()

    assert data.train_data.shape == (3949, 14)
    assert data.val_data.shape == (988, 14)
    assert data.test_data.shape == (1235, 14)

    assert data.train_data.shape[0] == data.train_labels.shape[0]
    assert data.val_data.shape[0] == data.val_labels.shape[0]
    assert data.test_data.shape[0] == data.test_labels.shape[0]

    assert all(data.train_labels.sum(axis=1) == 1)
    assert all(data.val_labels.sum(axis=1) == 1)
    assert all(data.test_labels.sum(axis=1) == 1)

    assert (
        data.train_data.columns.to_list()
        == data.val_data.columns.to_list()
        == data.test_data.columns.to_list()
    )

    assert any(data.train_data["priors_count"] <= 1)
    assert any(data.train_data["priors_count"] >= -1)

    assert "race+Other" in data.train_data.columns


def test_compas_no_val():
    data = Compas(val_prop=0)

    assert data.val_data is None
    assert data.val_labels is None

    assert data.train_data.shape == (4937, 14)
    assert data.test_data.shape == (1235, 14)

    assert data.train_data.shape[0] == data.train_labels.shape[0]
    assert data.test_data.shape[0] == data.test_labels.shape[0]

    assert all(data.train_labels.sum(axis=1) == 1)
    assert all(data.test_labels.sum(axis=1) == 1)

    assert (
        data.train_data.columns.to_list() == data.test_data.columns.to_list()
    )


def test_compas_no_test():
    data = Compas(test_prop=0)

    assert data.test_data is None
    assert data.test_labels is None

    assert data.train_data.shape == (4937, 14)
    assert data.val_data.shape == (1235, 14)

    assert data.train_data.shape[0] == data.train_labels.shape[0]
    assert data.val_data.shape[0] == data.val_labels.shape[0]

    assert all(data.train_labels.sum(axis=1) == 1)
    assert all(data.val_labels.sum(axis=1) == 1)

    assert data.train_data.columns.to_list() == data.val_data.columns.to_list()


def test_compas_groups():
    groups = {
        "race": {
            "Asian": "Others",
            "Hispanic": "Others",
            "Native American": "Others",
            "Other": "Others",
        }
    }

    data = Compas(groups=groups)

    assert all(
        data.train_data[
            ["race+African-American", "race+Caucasian", "race+Others"]
        ].sum(axis=1)
        == 1
    )
    assert all(
        data.val_data[
            ["race+African-American", "race+Caucasian", "race+Others"]
        ].sum(axis=1)
        == 1
    )
    assert all(
        data.test_data[
            ["race+African-American", "race+Caucasian", "race+Others"]
        ].sum(axis=1)
        == 1
    )

    assert data.train_data.shape == (3949, 11)
    assert data.val_data.shape == (988, 11)
    assert data.test_data.shape == (1235, 11)

    assert data.train_data.shape[0] == data.train_labels.shape[0]
    assert data.val_data.shape[0] == data.val_labels.shape[0]
    assert data.test_data.shape[0] == data.test_labels.shape[0]

    assert all(data.train_labels.sum(axis=1) == 1)
    assert all(data.val_labels.sum(axis=1) == 1)
    assert all(data.test_labels.sum(axis=1) == 1)

    assert (
        data.train_data.columns.to_list()
        == data.val_data.columns.to_list()
        == data.test_data.columns.to_list()
    )


def test_compas_num():
    data = Compas(features=["priors_count"])

    assert data.train_data.shape == (3949, 1)
    assert data.val_data.shape == (988, 1)
    assert data.test_data.shape == (1235, 1)

    assert data.train_data.shape[0] == data.train_labels.shape[0]
    assert data.val_data.shape[0] == data.val_labels.shape[0]
    assert data.test_data.shape[0] == data.test_labels.shape[0]

    assert all(data.train_labels.sum(axis=1) == 1)
    assert all(data.val_labels.sum(axis=1) == 1)
    assert all(data.test_labels.sum(axis=1) == 1)

    assert (
        data.train_data.columns.to_list()
        == data.val_data.columns.to_list()
        == data.test_data.columns.to_list()
    )


def test_compas_cat():
    data = Compas(features=["age_cat", "sex"])

    assert data.train_data.shape == (3949, 5)
    assert data.val_data.shape == (988, 5)
    assert data.test_data.shape == (1235, 5)

    assert data.train_data.shape[0] == data.train_labels.shape[0]
    assert data.val_data.shape[0] == data.val_labels.shape[0]
    assert data.test_data.shape[0] == data.test_labels.shape[0]

    assert all(data.train_labels.sum(axis=1) == 1)
    assert all(data.val_labels.sum(axis=1) == 1)
    assert all(data.test_labels.sum(axis=1) == 1)

    assert (
        data.train_data.columns.to_list()
        == data.val_data.columns.to_list()
        == data.test_data.columns.to_list()
    )


def test_compas_scaler():
    data = Compas(scaler=StandardScaler())

    assert data.train_data.shape == (3949, 14)
    assert data.val_data.shape == (988, 14)
    assert data.test_data.shape == (1235, 14)

    assert data.train_data.shape[0] == data.train_labels.shape[0]
    assert data.val_data.shape[0] == data.val_labels.shape[0]
    assert data.test_data.shape[0] == data.test_labels.shape[0]

    assert all(data.train_labels.sum(axis=1) == 1)
    assert all(data.val_labels.sum(axis=1) == 1)
    assert all(data.test_labels.sum(axis=1) == 1)

    assert (
        data.train_data.columns.to_list()
        == data.val_data.columns.to_list()
        == data.test_data.columns.to_list()
    )

    assert any(data.train_data["priors_count"] > 1)


def test_compas_prefix():
    data = Compas(prefix_sep="*")

    assert data.train_data.shape == (3949, 14)
    assert data.val_data.shape == (988, 14)
    assert data.test_data.shape == (1235, 14)

    assert data.train_data.shape[0] == data.train_labels.shape[0]
    assert data.val_data.shape[0] == data.val_labels.shape[0]
    assert data.test_data.shape[0] == data.test_labels.shape[0]

    assert all(data.train_labels.sum(axis=1) == 1)
    assert all(data.val_labels.sum(axis=1) == 1)
    assert all(data.test_labels.sum(axis=1) == 1)

    assert (
        data.train_data.columns.to_list()
        == data.val_data.columns.to_list()
        == data.test_data.columns.to_list()
    )

    assert "race*Other" in data.train_data.columns
