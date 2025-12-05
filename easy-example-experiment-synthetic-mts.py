#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from timeeval import (
    Algorithm,
    DatasetManager,
    DefaultMetrics,
    InputDimensionality,
    TimeEval,
    TrainingType,
)
from timeeval.adapters import FunctionAdapter  # for defining customized algorithm
from timeeval.algorithms import (cof, hbos, lof, cblof, random_black_forest, copod, torsk, autoencoder, dae, pcc,
                                 hbos_with_interpretability, cblof_with_interpretability, lof_with_interpretability,
                                    pcc_with_interpretability,copod_with_interpretability, cof_with_interpretability,
                                 random_black_forest_with_interpretability, dae_with_interpretability,
                                 autoencoder_with_interpretability)

from timeeval.data_types import AlgorithmParameter
from timeeval.metrics import RangePrVUS, RangeRocVUS
from timeeval.params import FixedParameters


def your_algorithm_function(
    data: AlgorithmParameter, args: Dict[str, Any]
) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.zeros_like(data)
    else:  # isinstance(data, pathlib.Path)
        return np.genfromtxt(data, delimiter=",", skip_header=1)[:,1:-1].mean(axis=1)


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    data_folder = cfg.data_folder
    custom_datasets_file = cfg.custom_datasets_file
    dm = DatasetManager(data_folder=Path(data_folder), custom_datasets_file= Path(custom_datasets_file) ,create_if_missing=True)
    datasets = dm.select()
    algorithms = [
        # list of algorithms which will be executed on the selected dataset(s)

        # cblof(params=FixedParameters({"random_state": 42})),
        # cof(params=FixedParameters({"n_neighbors": 20, "random_state": 42})),
        # lof(params=FixedParameters({"n_neighbors": 20, "random_state": 42})),
        # hbos(params=FixedParameters({"n_bin": 10, "random_state": 42})),
        # copod(params=FixedParameters({'random_state': 42})),
        # pcc(params=FixedParameters({'random_state': 42})),
        # torsk(params=FixedParameters({'random_state': 42})),

        # autoencoder(params=FixedParameters({'random_state': 42})),
        # dae(params=FixedParameters({'random_state': 42})),
        # random_black_forest(params=FixedParameters({'train_window_size': 24, 'random_state': 42})),

        cblof_with_interpretability(params=FixedParameters({"random_state": 42})),
        cof_with_interpretability(params=FixedParameters({"n_neighbors": 20, "random_state": 42})),
        lof_with_interpretability(params=FixedParameters({"n_neighbors": 20, "random_state": 42})),
        hbos_with_interpretability(params=FixedParameters({"n_bin": 10, "random_state": 42})),
        copod_with_interpretability(params=FixedParameters({'random_state': 42})),
        pcc_with_interpretability(params=FixedParameters({'random_state': 42})),
        # torsk_with_interpretability(params=FixedParameters({'random_state': 42})),

        autoencoder_with_interpretability(params=FixedParameters({'random_state': 42})),
        dae_with_interpretability(params=FixedParameters({'random_state': 42})),
        random_black_forest_with_interpretability(params=FixedParameters({'train_window_size': 24, 'random_state': 42})),


        # calling customized algorithm
        # Algorithm(
        #     name="MyPythonFunctionAlgorithm",
        #     main=FunctionAdapter(your_algorithm_function),
        #     data_as_file=True,
        #     training_type=TrainingType.UNSUPERVISED,
        #     input_dimensionality=InputDimensionality.MULTIVARIATE,
        # )
    ]

    timeeval = TimeEval(
        dm,
        datasets,
        algorithms,
        metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.RANGE_PR_AUC, DefaultMetrics.PR_AUC, RangePrVUS(max_buffer_size=100), RangeRocVUS(max_buffer_size=100)],
    )
    timeeval.run()
    results = timeeval.get_results(aggregated=False)
    print(results)


if __name__ == "__main__":
    main()
