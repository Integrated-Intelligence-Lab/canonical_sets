from sklearn.preprocessing import StandardScaler

from canonical_sets.data import Adult


def test_adult_no_preprocess():
    data = Adult(preprocess=False)

    assert data.train_data.shape == (32561, 15)
    assert data.test_data.shape == (16281, 15)


def test_adult():
    data = Adult()

    assert data.train_data.shape == (24129, 104)
    assert data.val_data.shape == (6033, 104)
    assert data.test_data.shape == (15060, 104)

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

    assert any(data.train_data["Age"] <= 1)
    assert any(data.train_data["Age"] >= -1)

    assert "Workclass+Private" in data.train_data.columns


def test_adult_no_val():
    data = Adult(val_prop=0)

    assert data.val_data is None
    assert data.val_labels is None

    assert data.train_data.shape == (30162, 104)
    assert data.test_data.shape == (15060, 104)

    assert data.train_data.shape[0] == data.train_labels.shape[0]
    assert data.test_data.shape[0] == data.test_labels.shape[0]

    assert all(data.train_labels.sum(axis=1) == 1)
    assert all(data.test_labels.sum(axis=1) == 1)

    assert (
        data.train_data.columns.to_list() == data.test_data.columns.to_list()
    )


def test_adult_groups():
    df = Adult(preprocess=False)

    others = list(df.train_data.Country.unique())
    others.remove("United-States")
    groups = {"Country": dict.fromkeys(others, "Others")}

    data = Adult(groups=groups)

    assert all(
        data.train_data[["Country+United-States", "Country+Others"]].sum(
            axis=1
        )
        == 1
    )
    assert all(
        data.val_data[["Country+United-States", "Country+Others"]].sum(axis=1)
        == 1
    )
    assert all(
        data.test_data[["Country+United-States", "Country+Others"]].sum(axis=1)
        == 1
    )

    assert data.train_data.shape == (24129, 65)
    assert data.val_data.shape == (6033, 65)
    assert data.test_data.shape == (15060, 65)

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


def test_adult_num():
    data = Adult(features=["Age", "Hours per week"])

    assert data.train_data.shape == (24129, 2)
    assert data.val_data.shape == (6033, 2)
    assert data.test_data.shape == (15060, 2)

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


def test_adult_cat():
    data = Adult(features=["Race", "Sex"])

    assert data.train_data.shape == (24129, 7)
    assert data.val_data.shape == (6033, 7)
    assert data.test_data.shape == (15060, 7)

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


def test_adult_scaler():
    data = Adult(scaler=StandardScaler())

    assert data.train_data.shape == (24129, 104)
    assert data.val_data.shape == (6033, 104)
    assert data.test_data.shape == (15060, 104)

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

    assert any(data.train_data.Age > 1)
    assert any(data.train_data.Age < -1)


def test_adult_prefix():
    data = Adult(prefix_sep="*")

    assert data.train_data.shape == (24129, 104)
    assert data.val_data.shape == (6033, 104)
    assert data.test_data.shape == (15060, 104)

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

    assert "Workclass*Private" in data.train_data.columns
