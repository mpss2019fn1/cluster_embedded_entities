from pathlib import Path
from typing import Iterable

from sklearn.cluster import DBSCAN

from clustering.abstract_cluster_builder import AbstractClusterBuilder

DBSCAN_METRIC = "euclidean"
# DBSCAN_METRIC = "cosine"


class DBScanClusterBuilder(AbstractClusterBuilder):

    def __init__(self, input_model_path: Path, parallel_executions: int, eps: float):
        AbstractClusterBuilder.__init__(self, input_model_path, parallel_executions)
        self._eps: float = eps

    def _train_specific_clusters(self) -> None:
        self._dbscan: DBSCAN = DBSCAN(algorithm='auto', eps=self._eps, metric=DBSCAN_METRIC,
                                      min_samples=3, n_jobs=self._parallel_executions)
        self._labels: Iterable[str] = self._dbscan.fit_predict(self._embeddings)

    def _map_embeddings_to_clusters(self) -> None:
        self._clusters = {label: [] for label in set(self._labels)}

        for i, word in enumerate(self._model.vocab):
            self._clusters[self._dbscan.labels_[i]].append(word)

    def name(self) -> str:
        return f"DBSCAN"
