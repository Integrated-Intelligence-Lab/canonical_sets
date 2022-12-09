"""LUCID-GAN."""
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from ctgan import CTGAN
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.ctgan import Discriminator, Generator
from torch import optim

from canonical_sets.gan.sampler import _Sampler


class LUCIDGAN(CTGAN):
    """Model wrapping `CTGAN` model.

    This class is based on the `CTGAN` class from the
    `ctgan` package. It has been modified to fix several bugs
    (see PRs on the `ctgan` GitHub page) and to allow for the
    extension of the conditional vector. Note that a part
    of the code and comments is identical to the original
    `CTGAN` class.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: Tuple[int, int] = (256, 256),
        discriminator_dim: Tuple[int, int] = (256, 256),
        generator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_lr: float = 2e-4,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        discriminator_steps: int = 1,
        log_frequency: bool = True,
        epochs: int = 300,
        pac: int = 10,
    ):
        """Initialize LUCIDGAN.

        Parameters
        ----------
        embedding_dim : int
            Size of the random noise passed to the generator. Defaults to 128.
        generator_dim : tuple of int
            Size of the output samples for each one of the residuals.
            A residual Layer will be created for each one of the values
            provided. Defaults to (256, 256).
        discriminator_dim : tuple of int
            Size of the output samples for each one of the discriminator
            layers. A linear Layer will be created for each one of the values
            provided. Defaults to (256, 256).
        generator_lr : float
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay : float
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr : float
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay : float
            Discriminator weight decay for the Adam Optimizer. Defaults to
            1e-6.
        batch_size : int
            Number of data samples to process in each step.
        discriminator_steps : int
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN
            implementation.
        log_frequency : bool
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        epochs : int
            Number of training epochs. Defaults to 300.
        pac : int
            Number of samples to group together when applying the
            discriminator. Defaults to 10.

        Attributes
        ----------
        generator_loss : list of torch.Tensor
            Generator loss at each epoch.
        reconsutrction_loss : list of torch.Tensor
            Reconstruction loss at each epoch.
        discriminator_loss : list of torch.Tensor
            Discriminator loss at each epoch.
        """
        super().__init__(
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            generator_decay=generator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_decay=discriminator_decay,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            log_frequency=log_frequency,
            verbose=False,
            epochs=epochs,
            pac=pac,
            cuda=False,
        )

        self.generator_loss: List[torch.Tensor] = []
        self.reconstruction_loss: List[torch.Tensor] = []
        self.discriminator_loss: List[torch.Tensor] = []

    @random_state
    def fit(
        self,
        train_data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
        conditional: Optional[List[str]] = None,
    ):
        """Fit LUCID-GAN to the training data.

        Parameters
        ----------
        train_data : pandas.DataFrame
            Training Data. It must be a 2-dimensional pandas.DataFrame where
            each column is a feature and each row is a sample.
        discrete_columns : list of str, optional
            List of discrete columns to be used to create the conditional
            vector. This list should contain the column names. Note that if
            ``None``, we select all columns which dtype is not ``number``.
            See pd.DataFrame.select_dtypes for more information.
        conditional : list of str, optional
            List of columns with the conditional features which should not
            be part of the generated data. Note that the columns in this list
            should be ``numeric``, and not be included in the
            ``discrete_columns``.
        """
        self._n_conditions: Optional[int] = None
        self._conditions: Optional[np.ndarray] = None
        self._conditions_columns: Optional[List[str]] = None

        if conditional is not None:
            self._n_conditions = len(conditional)
            self._conditions = train_data[conditional].to_numpy()
            self._conditions_columns = conditional

            train_data = train_data.drop(conditional, axis=1)

        if discrete_columns is None:
            discrete_columns = train_data.select_dtypes(
                exclude=["number"]
            ).columns.tolist()

        self._validate_discrete_columns(train_data, discrete_columns)

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        data_dim = self._transformer.output_dimensions

        self._data_sampler = _Sampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency,
            self._conditions,
        )

        if self._conditions is None:
            self._generator = Generator(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim,
            )

            discriminator = Discriminator(
                data_dim + self._data_sampler.dim_cond_vec(),
                self._discriminator_dim,
                pac=self.pac,
            )

        else:
            self._generator = Generator(
                self._embedding_dim
                + self._data_sampler.dim_cond_vec()
                + self._n_conditions,
                self._generator_dim,
                data_dim,
            )

            discriminator = Discriminator(
                data_dim
                + self._data_sampler.dim_cond_vec()
                + self._n_conditions,
                self._discriminator_dim,
                pac=self.pac,
            )

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        with tqdm.trange(self._epochs) as epochs_bar:
            for i in epochs_bar:
                for id_ in range(steps_per_epoch):

                    # discriminator training loop
                    for n in range(self._discriminator_steps):

                        # generate noise for fake data
                        fakez = torch.normal(mean=mean, std=std)

                        # if no discrete columns, all are None
                        condvec = self._data_sampler.sample_condvec(
                            self._batch_size
                        )  # Dict with c1, m1, col, opt

                        # if condvec is None, set all of its elements to None
                        if condvec is None:

                            # if conditions are not provided, sample both
                            # real data and conditions and create a random
                            # permutation of the conditions for the fake
                            # data generation and add it to the noise
                            if self._conditions is not None:
                                (
                                    real,
                                    conditions_real,
                                ) = self._data_sampler.sample_data(
                                    self._batch_size
                                )

                                perm = np.arange(self._batch_size)
                                np.random.shuffle(perm)

                                conditions_fake = conditions_real[perm, :]

                                fakez = torch.cat(
                                    [fakez, conditions_fake], dim=1
                                )

                            # else, only sample real data unconditionally,
                            # i.e. no conditional vector and conditions
                            else:
                                real = self._data_sampler.sample_data(
                                    self._batch_size
                                )

                        # if discrete columns are provided, sample
                        # conditional vectors and create a random
                        # permutation of the conditional vectors for
                        # the fake data generation
                        else:
                            c1, m1, col, opt = condvec
                            c1 = torch.from_numpy(c1)
                            m1 = torch.from_numpy(m1)

                            fakez = torch.cat([fakez, c1], dim=1)

                            perm = np.arange(self._batch_size)
                            np.random.shuffle(perm)

                            c2 = c1[perm]

                            # if there are no conditions, sample only real
                            # data, conditional on the permutated conditional
                            # vector
                            if self._conditions is None:
                                real = self._data_sampler.sample_data(
                                    self._batch_size, col[perm], opt[perm]
                                )

                            # else sample both real data and conditions,
                            # conditional on the conditional vector and
                            # create the same permutation of the conditions
                            # (as for the conditional vector) for the fake
                            # data generation
                            else:
                                (
                                    real,
                                    conditions,
                                ) = self._data_sampler.sample_data(
                                    self._batch_size, col[perm], opt[perm]
                                )

                                conditions_c2 = conditions
                                conditions_c1 = conditions_c2[
                                    np.argsort(perm), :
                                ]

                                fakez = torch.cat(
                                    [fakez, conditions_c1], dim=1
                                )

                        fake = self._generator(fakez)
                        fakeact = self._apply_activate(fake)

                        if condvec is None:
                            if self._conditions is None:
                                real_cat = real
                                fake_cat = fakeact

                            else:
                                real_cat = torch.cat(
                                    [real, conditions_real], dim=1
                                )
                                fake_cat = torch.cat(
                                    [fakeact, conditions_fake], dim=1
                                )

                        else:
                            if self._conditions is None:
                                real_cat = torch.cat([real, c2], dim=1)
                                fake_cat = torch.cat([fakeact, c1], dim=1)

                            else:
                                real_cat = torch.cat(
                                    [real, c2, conditions_c2], dim=1
                                )
                                fake_cat = torch.cat(
                                    [fakeact, c1, conditions_c1], dim=1
                                )

                        y_fake = discriminator(fake_cat)
                        y_real = discriminator(real_cat)

                        pen = discriminator.calc_gradient_penalty(
                            real_cat, fake_cat, pac=self.pac
                        )
                        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                        optimizerD.zero_grad()
                        pen.backward(retain_graph=True)
                        loss_d.backward()
                        optimizerD.step()

                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._data_sampler.sample_condvec(
                        self._batch_size
                    )

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None

                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1)
                        m1 = torch.from_numpy(m1)
                        fakez = torch.cat([fakez, c1], dim=1)

                    if self._conditions is not None:
                        real, conditions = self._data_sampler.sample_data(
                            self._batch_size, col, opt
                        )

                        fakez = torch.cat([fakez, conditions], dim=1)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if condvec is None:
                        if self._conditions is None:
                            y_fake = discriminator(fakeact)

                        else:
                            y_fake = discriminator(
                                torch.cat([fakeact, conditions], dim=1)
                            )

                    else:
                        if self._conditions is None:
                            y_fake = discriminator(
                                torch.cat([fakeact, c1], dim=1)
                            )

                        else:
                            y_fake = discriminator(
                                torch.cat([fakeact, c1, conditions], dim=1)
                            )

                    if condvec is None:
                        cross_entropy = torch.zeros(1)

                    else:
                        cross_entropy = self._cond_loss(fake, c1, m1)

                    loss_g = -torch.mean(y_fake) + cross_entropy

                    optimizerG.zero_grad()
                    loss_g.backward()
                    optimizerG.step()

                epochs_bar.set_description(
                    f"Epoch {i}, "
                    f"Loss G: {-torch.mean(y_fake).detach().cpu(): .4f}, "
                    f"Loss R: {cross_entropy.detach().cpu().item(): .4f}, "
                    f"Loss D: {loss_d.detach().cpu(): .4f}"
                )

                self.generator_loss.append(loss_g.detach().cpu())
                self.reconstruction_loss.append(cross_entropy.detach().cpu())
                self.discriminator_loss.append(loss_d.detach().cpu())

    @random_state
    def sample(
        self,
        n: int,
        condition_column: Optional[Union[List[str], str]] = None,
        condition_value: Optional[Union[List[str], str]] = None,
        conditional: Optional[pd.DataFrame] = None,
        empirical: bool = False,
    ):
        """Sample data from LUCIDGAN.

        Choosing a condition_column and condition_value will increase the
        probability of the discrete condition_value happening in the
        condition_column.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        condition_column : str or list of str, optional
            Name(s) of the discrete column(s) to condition on. Both
            condition_column and condition_value must be specified to
            condition on a discrete column. Otherwise this argument is ignored.
        condition_value : str or list of str, optional
            Name(s) of the category in the condition_column(s) which we wish to
            increase the probability of happening. Both condition_value
            and condition_column must be specified to condition on a discrete
            column. Otherwise this argument is ignored.
        conditional : pandas.DataFrame, optional
            A 2-dimensional ``pandas.DataFrame`` with ``numeric`` values
            for the conditional features. The number of rows of the 2d array
            should be the same as ``n`` or be equal to one (in which case it
            will be repeated ``n`` times). The number of columns should be
            equal to the number of conditional features.
        empirical : bool
            Whether to use the empirical distribution of the data to sample
            from. If False, the generator will be used to sample from the
            learned distribution. Default is False.

        Returns
        -------
        pandas.DataFrame
            A ``pandas.DataFrame`` with ``n`` samples.
        """
        self._generator.eval()

        if condition_column is not None and condition_value is not None:

            if isinstance(condition_column, str) and isinstance(
                condition_value, str
            ):
                condition_column = [condition_column]
                condition_value = [condition_value]

            condition_info = self._convert_column_name_value_to_id(
                condition_column, condition_value  # type: ignore
            )
            global_condition_vec = (
                self._data_sampler.generate_cond_from_condition_column_info(
                    condition_info, self._batch_size
                )
            )
        else:
            global_condition_vec = None

        if conditional is not None and self._conditions is not None:
            if not all(
                conditional.apply(
                    lambda s: pd.to_numeric(s, errors="coerce").notnull().all()
                )
            ):
                raise ValueError("The conditional data must be numeric.")

            conditional = torch.from_numpy(
                conditional.to_numpy().astype("float32")
            )

            if conditional.shape[0] == 1:
                conditional = conditional.repeat(self._batch_size, 1)

            elif conditional.shape[0] != n:
                raise ValueError(
                    "conditional must have either length one or n."
                )

        elif conditional is None and self._conditions is not None:
            cond_len = self._conditions.shape[0]

        elif conditional is not None and self._conditions is None:
            warnings.warn(
                "self._conditions is None so the specified conditional "
                "argument will be ignored."
            )

        steps = n // self._batch_size + 1
        data = []

        if self._conditions is not None:
            data_cond = []

        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()

                if conditional is not None and self._conditions is not None:
                    conditions = conditional

                elif conditional is None and self._conditions is not None:
                    conditions = self._conditions[
                        np.random.randint(cond_len, size=self._batch_size)
                    ]

            else:
                if conditional is not None and self._conditions is not None:

                    if empirical:
                        (
                            condvec,
                            _,
                        ) = self._data_sampler.sample_original_condvec(
                            self._batch_size
                        )

                    else:
                        condvec = np.zeros(
                            (
                                self._batch_size,
                                self._data_sampler.dim_cond_vec(),
                            ),
                            dtype=np.float32,
                        )

                    conditions = conditional

                elif conditional is None and self._conditions is not None:

                    if empirical:
                        (
                            condvec,
                            conditions,
                        ) = self._data_sampler.sample_original_condvec(
                            self._batch_size
                        )

                    else:
                        (
                            _,
                            conditions,
                        ) = self._data_sampler.sample_original_condvec(
                            self._batch_size
                        )
                        condvec = np.zeros(
                            (
                                self._batch_size,
                                self._data_sampler.dim_cond_vec(),
                            ),
                            dtype=np.float32,
                        )

                else:
                    if empirical:
                        condvec = self._data_sampler.sample_original_condvec(
                            self._batch_size
                        )  # type: ignore

                    else:
                        condvec = np.zeros(
                            (
                                self._batch_size,
                                self._data_sampler.dim_cond_vec(),
                            ),
                            dtype=np.float32,
                        )

            if condvec is not None:
                c1 = torch.from_numpy(condvec)
                fakez = torch.cat([fakez, c1], dim=1)

            if self._conditions is not None:
                fakez = torch.cat([fakez, conditions], dim=1)

            with torch.no_grad():
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

            data.append(fakeact.detach().cpu().numpy())

            if self._conditions is not None:
                data_cond.append(conditions.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        if self._conditions is not None:
            data_cond = np.concatenate(data_cond, axis=0)
            data_cond = data_cond[:n]

        if self._conditions is not None:
            transformed_data = self._transformer.inverse_transform(data)
            transformed_cond = pd.DataFrame(
                data_cond, columns=self._conditions_columns
            )

            return pd.concat([transformed_data, transformed_cond], axis=1)
        else:
            return self._transformer.inverse_transform(data)

    def _convert_column_name_value_to_id(
        self, column_names: List[str], values: List[str]
    ) -> List[Dict[str, int]]:
        """Get the ids of the given `column_name`.

        Parameters
        ----------
        column_names : List[str]
            The column names.
        values : List[str]
            The values of the columns.

        Returns
        -------
        List[Dict[str, int]]
            The ids of the given `column_name`.
        """
        results = []

        for column_name, value in zip(column_names, values):
            discrete_counter = 0
            column_id = 0
            for (
                column_transform_info
            ) in self._transformer._column_transform_info_list:
                if column_transform_info.column_name == column_name:
                    break
                if column_transform_info.column_type == "discrete":
                    discrete_counter += 1

                column_id += 1

            else:
                raise ValueError(
                    f"The column_name `{column_name}` "
                    f"doesn't exist in the data."
                )

            ohe = column_transform_info.transform
            data = pd.DataFrame(
                [value], columns=[column_transform_info.column_name]
            )
            one_hot = ohe.transform(data).to_numpy()[0]
            if sum(one_hot) == 0:
                raise ValueError(
                    f"The value `{value}` doesn't exist in the"
                    f"column `{column_name}`."
                )

            results.append(
                {
                    "discrete_column_id": discrete_counter,
                    "column_id": column_id,
                    "value_id": np.argmax(one_hot),
                }
            )

        return results  # type: ignore
