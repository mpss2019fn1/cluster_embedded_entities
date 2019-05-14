from sklearn.cluster import DBSCAN

from clusterers.abstract_clusterer import AbstractClusterer

DBSCAN_METRIC = "euclidean"
# DBSCAN_METRIC = "cosine"


class DBScanClusterer(AbstractClusterer):

    def __init__(self, input_model_path, parallel_executions, eps):
        AbstractClusterer.__init__(self, input_model_path, parallel_executions)
        self._eps = eps

    def _train_specific_clusters(self):
        self._dbscan = DBSCAN(algorithm='auto', eps=self._eps, metric=DBSCAN_METRIC,
                              min_samples=3, n_jobs=self._parallel_executions)
        self._labels = self._dbscan.fit_predict(self._embeddings)

    def _map_embeddings_to_clusters(self):
        self._clusters = {k: [] for k in set(self._labels)}

        for i, word in enumerate(self._model.docvecs.doctags):
            self._clusters[self._dbscan.labels_[i]].append(word)

        return self._clusters

    def name(self):
        return f"DBSCAN"
