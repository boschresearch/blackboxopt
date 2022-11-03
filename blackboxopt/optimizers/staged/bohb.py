# Copyright (c) 2020 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/boschresearch/blackboxopt
#
# SPDX-License-Identifier: Apache-2.0
#
# This source code is derived from HpBandSter 0.7.4
#  https://github.com/automl/HpBandSter
# Copyright (c) 2017-2018, ML4AAD, licensed under the BSD-3 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
# Changes include:
#     - integration into the blackboxopt API
#     - move code into functions instead of class methods
#     - docstrings and type hints

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import parameterspace as ps
import scipy.stats as sps
import statsmodels.api as sm
from parameterspace import ParameterSpace

from blackboxopt import Evaluation, Objective
from blackboxopt.optimizers.staged.configuration_sampler import (
    StagedIterationConfigurationSampler,
)


def sample_around_values(
    datum: np.ndarray,
    bandwidths: np.ndarray,
    vartypes: Union[list, np.ndarray],
    min_bandwidth: float,
    bw_factor: float,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Sample numerical representation close to a given datum.

    This is specific to the KDE in statsmodels and their kernel for the different
    variable types.

    Args:
        datum: Numerical representation of a configuration that is used as the 'center'
            for sampling.
        bandwidths: Bandwidth of the corresponding kernels in each dimension.
        vartypes: Encoding of the types of the variables: 0 mean continuous, >0 means
            categorical with as many different values.
        min_bandwidth: Smallest allowed bandwidth. Ensures diversity even if all
            samples agree on a value in a dimension.
        bw_factor: To increase diversity, the bandwidth is actually multiplied by this
            factor before sampling.
        rng: A random number generator to make the sampling reproducible.

    Returns:
        Numerical representation of a configuration close to the provided datum.
    """
    rng = np.random.default_rng(rng)

    vector = []
    for m, bw, t in zip(datum, bandwidths, vartypes):
        bw = max(bw, min_bandwidth)
        if t == 0:
            bw = bw_factor * bw
            try:
                v = sps.truncnorm.rvs(
                    -m / bw, (1 - m) / bw, loc=m, scale=bw, random_state=rng
                )
            except Exception:
                return None
        elif t > 0:
            v = m if rng.random() < (1 - bw) else rng.integers(t)
        else:
            bw = min(0.9999, bw)  # bandwidth has to be less the one for this kernel!
            diffs = np.abs(np.arange(-t) - m)
            probs = 0.5 * (1 - bw) * (bw**diffs)
            idx = diffs == 0
            probs[idx] = (idx * (1 - bw))[idx]
            probs /= probs.sum()
            v = rng.choice(-t, p=probs)
        vector.append(v)
    return np.array(vector)


def convert_to_statsmodels_kde_representation(
    array: np.ndarray, vartypes: Union[list, np.ndarray]
) -> np.ndarray:
    """Convert numerical representation for categoricals and ordinals to integers.
    Args:
        array: Numerical representation of the configurations with categorical and ordinal values
            mapped into the unit hypercube.
        vartypes: Encoding of the types of the variables: 0 mean continuous, >0 means
            categorical with as many different values, and <0 means ordinal with as many values.

    Returns:
        Numerical representation consistent with the statsmodels package.
    """
    processed_vector = np.copy(array)

    for i in range(len(processed_vector)):
        if vartypes[i] == 0:
            continue
        num_values = abs(vartypes[i])
        processed_vector[i] = np.around((processed_vector[i] * num_values) - 0.5)

    return processed_vector


def convert_from_statsmodels_kde_representation(
    array: np.ndarray, vartypes: Union[list, np.ndarray]
) -> np.ndarray:
    """Convert numerical representation for categoricals and ordinals back into the unit hypercube.

    Args:
        array: Numerical representation of the configurations following the statsmodels convention for
            categorical and ordinal values being integers.
        vartypes: Encoding of the types of the variables: 0 mean continuous, >0 means
            categorical with as many different values, and <0 means ordinal with as many values.

    Returns:
        Numerical representation consistent with a numerical representation in the hypercube.
    """
    processed_vector = np.copy(array)

    for i in range(len(processed_vector)):
        if vartypes[i] != 0:
            num_values = abs(vartypes[i])
            processed_vector[i] = (processed_vector[i] + 0.5) / num_values

    return processed_vector


def impute_conditional_data(
    array: np.ndarray,
    vartypes: Union[list, np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Impute NaNs in numerical representation with observed values or prior samples.

    This method is needed to use the `statsmodels` KDE, which doesn't handle missing
    values out of the box.

    Args:
        array: Numerical representation of the configurations which can include NaN
            values for inactive variables.
        vartypes: Encoding of the types of the variables: 0 mean continuous, >0 means
            categorical with as many different values, and <0 means ordinal with as many values.
        rng: A random number generator to make the imputation reproducible.
    Returns:
        Numerical representation where all NaNs have been replaced with observed values
        or prior samples.
    """
    rng = np.random.default_rng(rng)

    return_array = np.empty_like(array)

    for i in range(array.shape[0]):
        datum = np.copy(array[i])
        nan_indices = np.argwhere(np.isnan(datum)).flatten()

        while np.any(nan_indices):
            nan_idx = nan_indices[0]
            valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

            if len(valid_indices) > 0:
                # pick one of them at random and overwrite all NaN values
                row_idx = rng.choice(valid_indices)
                datum[nan_indices] = array[row_idx, nan_indices]

            else:
                # no point in the data has this value activated, so fill it with a valid
                # but random value
                t = vartypes[nan_idx]
                if t == 0:
                    datum[nan_idx] = rng.random()
                elif t > 0:
                    datum[nan_idx] = rng.integers(t)
                elif t < 0:
                    datum[nan_idx] = rng.integers(-t)
            nan_indices = np.argwhere(np.isnan(datum)).flatten()
        return_array[i, :] = datum
    return return_array


class Sampler(StagedIterationConfigurationSampler):
    def __init__(
        self,
        search_space: ParameterSpace,
        objective: Objective,
        min_samples_in_model: int,
        top_n_percent: int,
        num_samples: int,
        random_fraction: float,
        bandwidth_factor: float,
        min_bandwidth: float,
        seed: int = None,
        logger=None,
    ):
        """Fits for each given fidelity a kernel density estimator on the best N percent
        of the evaluated configurations on this fidelity.

        Args:
            search_space: ConfigurationSpace/ ParameterSpace object.
            objective: The objective of the optimization.
            min_samples_in_model: Minimum number of datapoints needed to fit a model.
            top_n_percent: Determines the percentile of configurations that will be used
                as training data for the kernel density estimator of the good
                configuration, e.g if set to 10 the best 10% configurations will be
                considered for training.
            num_samples: Number of samples drawn to optimize EI via sampling.
            random_fraction: Fraction of random configurations returned
            bandwidth_factor: Widens the bandwidth for contiuous parameters for
                proposed points to optimize EI
            min_bandwidth: To keep diversity, even when all (good) samples have the
                same value for one of the parameters, a minimum bandwidth
                (reasonable default: 1e-3) is used instead of zero.
            seed: A seed to make the sampler reproducible.
            logger: [description]

        Raises:
            RuntimeError: [description]
        """
        self.logger = logging.getLogger("blackboxopt") if logger is None else logger

        self.objective = objective
        self.min_samples_in_model = min_samples_in_model
        self.top_n_percent = top_n_percent
        self.search_space = search_space
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        if self.min_samples_in_model < len(search_space) + 1:
            self.min_samples_in_model = len(search_space) + 1
            self.logger.warning(
                "Invalid min_samples_in_model value. "
                + f"Setting it to {self.min_samples_in_model}"
            )

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        self.kde_vartypes = ""

        vartypes: List[Union[float, int]] = []
        for hp in search_space:  # type: ignore
            hp = hp["parameter"]
            if isinstance(hp, (ps.ContinuousParameter, ps.IntegerParameter)):
                self.kde_vartypes += "c"
                vartypes.append(0)

            elif isinstance(hp, ps.CategoricalParameter):
                self.kde_vartypes += "u"
                vartypes.append(hp.num_values)

            elif isinstance(hp, ps.OrdinalParameter):
                self.kde_vartypes += "o"
                vartypes.append(-hp.num_values)
            else:
                raise RuntimeError(f"This version on BOHB does not support {type(hp)}!")

        self.vartypes = np.array(vartypes, dtype=int)

        self.configs: Dict[float, List[np.ndarray]] = dict()
        self.losses: Dict[float, List[float]] = dict()
        self.kde_models: Dict[float, dict] = dict()

    def sample_configuration(self) -> Tuple[dict, dict]:
        """[summary]

        Returns:
            [description]
        """
        self.logger.debug("start sampling a new configuration.")

        # Sample from prior, if no model is available or with given probability
        if len(self.kde_models) == 0 or self._rng.random() < self.random_fraction:
            return self.search_space.sample(), {"model_based_pick": False}

        best = np.inf
        best_vector = None

        try:
            # sample from largest fidelity
            fidelity = max(self.kde_models.keys())

            good = self.kde_models[fidelity]["good"].pdf
            bad = self.kde_models[fidelity]["bad"].pdf

            minimize_me = lambda x: max(1e-32, bad(x)) / max(good(x), 1e-32)

            kde_good = self.kde_models[fidelity]["good"]
            kde_bad = self.kde_models[fidelity]["bad"]

            for i in range(self.num_samples):
                idx = self._rng.integers(0, len(kde_good.data))
                datum = kde_good.data[idx]
                vector = sample_around_values(
                    datum,
                    kde_good.bw,
                    self.vartypes,
                    self.min_bandwidth,
                    self.bw_factor,
                    rng=self._rng,
                )
                if vector is None:
                    continue

                # Statsmodels KDE estimators relies on seeding through numpy's global
                # state. We do this close to the evaluation of the PDF (`good`, `bad`)
                # to increase robustness for multi threading.
                # As we seed in a loop, we need to change it each iteration to not get
                # the same random numbers each time.
                # We also reset the np.random's global state, in case the user relies
                # on it in other parts of the code and to not hide other determinism
                # issues.
                # TODO: Check github issue if there was progress and the seeding can be
                # removed: https://github.com/statsmodels/statsmodels/issues/306
                cached_rng_state = None
                if self.seed:
                    cached_rng_state = np.random.get_state()
                    np.random.seed(self.seed + i)

                val = minimize_me(vector)

                if cached_rng_state:
                    np.random.set_state(cached_rng_state)

                if not np.isfinite(val):
                    self.logger.warning(
                        "sampled vector: %s has EI value %s" % (vector, val)
                    )
                    self.logger.warning(
                        "data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data)
                    )
                    self.logger.warning(
                        "bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw)
                    )

                    # right now, this happens because a KDE does not contain all values
                    # for a categorical parameter this cannot be fixed with the
                    # statsmodels KDE, so for now, we are just going to evaluate this
                    # one if the good_kde has a finite value, i.e. there is no config
                    # with that value in the bad kde, so it shouldn't be terrible.
                    if np.isfinite(good(vector)) and best_vector is not None:
                        best_vector = vector
                    continue

                if val < best:
                    best = val
                    best_vector = convert_from_statsmodels_kde_representation(
                        vector, self.vartypes
                    )

            if best_vector is None:
                self.logger.debug(
                    f"Sampling based optimization with {self.num_samples} samples did "
                    + "not find any finite/numerical acquisition function value "
                    + "-> using random configuration"
                )
                return self.search_space.sample(), {"model_based_pick": False}
            else:
                self.logger.debug(
                    "best_vector: {}, {}, {}, {}".format(
                        best_vector, best, good(best_vector), bad(best_vector)
                    )
                )
                return (
                    self.search_space.from_numerical(best_vector),
                    {"model_based_pick": True},
                )

        except Exception as e:
            self.logger.debug(
                "Sample base optimization failed. Falling back to a random sample."
            )
            return self.search_space.sample(), {"model_based_pick": False}

    def digest_evaluation(self, evaluation: Evaluation):
        """[summary]

        Args:
            evaluation: [description]
        """
        objective_value = evaluation.objectives[self.objective.name]
        if objective_value is None:
            loss = np.inf
        else:
            loss = (
                -objective_value
                if self.objective.greater_is_better
                else objective_value
            )
        config_vector = self.search_space.to_numerical(evaluation.configuration)
        config_vector = convert_to_statsmodels_kde_representation(
            config_vector, self.vartypes
        )

        fidelity = evaluation.settings["fidelity"]

        if fidelity not in self.configs.keys():
            self.configs[fidelity] = []
            self.losses[fidelity] = []

        self.configs[fidelity].append(config_vector)
        self.losses[fidelity].append(loss)

        if bool(self.kde_models.keys()) and max(self.kde_models.keys()) > fidelity:
            return

        if np.isfinite(self.losses[fidelity]).sum() <= self.min_samples_in_model - 1:
            n_runs_finite_loss = np.isfinite(self.losses[fidelity]).sum()
            self.logger.debug(
                f"Only {n_runs_finite_loss} run(s) with a finite loss for fidelity "
                + f"{fidelity} available, need more than {self.min_samples_in_model+1} "
                + "-> can't build model!"
            )
            return

        train_configs = np.array(self.configs[fidelity])
        train_losses = np.array(self.losses[fidelity])

        n_good = max(
            self.min_samples_in_model,
            (self.top_n_percent * train_configs.shape[0]) // 100,
        )

        n_bad = max(
            self.min_samples_in_model,
            ((100 - self.top_n_percent) * train_configs.shape[0]) // 100,
        )

        # Refit KDE for the current fidelity
        idx = np.argsort(train_losses)

        train_data_good = impute_conditional_data(
            train_configs[idx[:n_good]], self.vartypes, rng=self._rng
        )
        train_data_bad = impute_conditional_data(
            train_configs[idx[n_good : n_good + n_bad]], self.vartypes, rng=self._rng
        )

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        # more expensive crossvalidation method
        # bw_estimation = 'cv_ls'
        # quick rule of thumb
        bw_estimation = "normal_reference"

        bad_kde = sm.nonparametric.KDEMultivariate(
            data=train_data_bad,
            var_type=self.kde_vartypes,
            bw=bw_estimation,
        )
        good_kde = sm.nonparametric.KDEMultivariate(
            data=train_data_good,
            var_type=self.kde_vartypes,
            bw=bw_estimation,
        )

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde_models[fidelity] = {"good": good_kde, "bad": bad_kde}

        # update probs for the categorical parameters for later sampling
        self.logger.debug(
            f"done building a new model for fidelity {fidelity} based on "
            + f"{n_good}/{n_bad} split\nBest loss for this fidelity: "
            + f"{np.min(train_losses)}\n"
            + ("=" * 40)
        )
