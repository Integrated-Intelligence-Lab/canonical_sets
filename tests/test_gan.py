import pandas as pd

from canonical_sets import LUCIDGAN
from canonical_sets.data import Adult


def test_dis_cond():
    adult = Adult()

    train_data = adult.inverse_preprocess(adult.train_data)
    train_data = train_data[["Age", "Hours per week", "Education"]]

    discrete_columns = ["Education"]
    conditional = ["Hours per week"]

    lucidgan = LUCIDGAN(epochs=1)
    lucidgan.fit(train_data, discrete_columns, conditional)

    lucidgan.sample(
        n=10,
        condition_column="Education",
        condition_value="HS-grad",
        conditional=pd.DataFrame({"Hours per week": [40]}),
    )


def test_cond():
    adult = Adult()

    train_data = adult.inverse_preprocess(adult.train_data)
    train_data = train_data[["Age", "Hours per week"]]

    conditional = ["Hours per week"]

    lucidgan = LUCIDGAN(epochs=1)
    lucidgan.fit(train_data, conditional=conditional)

    lucidgan.sample(
        n=10,
        conditional=pd.DataFrame({"Hours per week": [40]}),
    )


def test_dis():
    adult = Adult()

    train_data = adult.inverse_preprocess(adult.train_data)
    train_data = train_data[["Age", "Hours per week", "Education"]]

    discrete_columns = ["Education"]

    lucidgan = LUCIDGAN(epochs=1)
    lucidgan.fit(train_data, discrete_columns)

    lucidgan.sample(
        n=10,
        condition_column="Education",
        condition_value="HS-grad",
    )


def test_gan():
    adult = Adult()

    train_data = adult.inverse_preprocess(adult.train_data)
    train_data = train_data[["Age", "Hours per week"]]

    lucidgan = LUCIDGAN(epochs=1)
    lucidgan.fit(train_data)

    lucidgan.sample(n=10)
