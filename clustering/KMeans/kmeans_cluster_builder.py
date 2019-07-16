from pathlib import Path

from sklearn import cluster

from clustering.abstract_cluster_builder import AbstractClusterBuilder


class KMeansClusterBuilder(AbstractClusterBuilder):

    def __init__(self, input_model_path: Path, parallel_executions: int, k: int):
        AbstractClusterBuilder.__init__(self, input_model_path, parallel_executions)
        self._k: int = k

    def _train_specific_clusters(self) -> None:
        self._kmeans: cluster.KMeans = cluster.KMeans(n_clusters=self._k,
                                                      algorithm="auto",
                                                      init="k-means++",
                                                      n_jobs=self._parallel_executions)
        self._kmeans = self._kmeans.fit(self._embeddings)

    def _map_embeddings_to_clusters(self) -> None:
        self._clusters = {k: [] for k in range(self._k)}

        for i, word in enumerate(self._model.docvecs.doctags):
            self._clusters[self._kmeans.labels_[i]].append(word)

    def name(self) -> str:
        return f"k-means-{self._k}"
