import tempfile
import os
from abc import ABC
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from .docker import DockerAdapter
from ..data_types import AlgorithmParameter
import numpy as np
import pandas as pd

class InterpretabilityAdapter(DockerAdapter):

    def __init__(
        self, adapter: DockerAdapter, top_k: int
    ) -> None:
        self._adapter = adapter
        self.top_k = top_k

    def _get_anomaly_scores_per_var(self, dataset: AlgorithmParameter)-> pd.DataFrame:
        df: Optional[pd.DataFrame] = None
        if isinstance(dataset, Path):
            df = pd.read_csv(dataset)
        elif isinstance(dataset, np.ndarray):
            df = pd.DataFrame(dataset)
        else:
            raise ValueError(f"Dataset must be either a path or numpy array")
        return df

    def _call(
            self, dataset: AlgorithmParameter, args: Dict[str, Any]
    ) -> AlgorithmParameter:

        with tempfile.TemporaryDirectory() as tmp_dir:
            anomaly_scores_per_var_path = os.path.join(tmp_dir, "scores_per_var.csv")
            accuracy_scores = self._adapter._call(dataset, args)
            print("accuracy SCORES shape", accuracy_scores.shape)
            anomaly_scores_per_var = self._adapter._read_results_per_var(args)
            multivariate_labels = self._adapter._read_multivariate_labels(args)
            if anomaly_scores_per_var is not None:
                print("anomaly_scores_per_var. shape", anomaly_scores_per_var.shape)
            if multivariate_labels is not None:
                print("multivariate labels. shape", multivariate_labels.shape)

            # multivariate_labels = multivariate_labels + 1
            # multivariate_labels = multivariate_labels/multivariate_labels.sum(axis=1,  keepdims=True)
            # anomaly_scores_per_var = anomaly_scores_per_var/anomaly_scores_per_var.sum(axis=1,  keepdims=True)
            anomaly_scores_per_var_ranking = np.argsort(anomaly_scores_per_var, axis=1)
            top_1_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
            interpretability_list = []
            for labels, top_1_index in zip(multivariate_labels, top_1_anomalous_dimension):
                if labels.sum() != 0.0:
                    interpretability = labels[top_1_index].sum()/labels.sum()
                    interpretability_list.append(interpretability)
                else:
                    interpretability_list.append(0.0)
            # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
            interpretability_scores = np.array(interpretability_list)
            return interpretability_scores

            # anomaly_scores_per_var_df = pd.read_csv(anomaly_scores_per_var_path)
            # return anomaly_scores_per_var_df.max(axis=1).values


    # def _read_results(self, args: Dict[str, Any]) -> np.ndarray:
    #     results: np.ndarray = np.genfromtxt(
    #         self._results_path(args) / SCORES_FILE_NAME, delimiter=","
    #     )
    #     return results

    def get_prepare_fn(self) -> Optional[Callable[[], None]]:
        return self._adapter.get_prepare_fn()

    def get_finalize_fn(self) -> Optional[Callable[[], None]]:
        return self._adapter.get_finalize_fn()