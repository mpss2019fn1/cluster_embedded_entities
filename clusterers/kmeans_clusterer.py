from sklearn import cluster

from clusterers.abstract_clusterer import AbstractClusterer


class KMeansClusterer(AbstractClusterer):

    def __init__(self, input_model_path, parallel_executions, k):
        AbstractClusterer.__init__(self, input_model_path, parallel_executions)
        self._k = k

    def _train_specific_clusters(self):
        self._kmeans = cluster.KMeans(n_clusters=int(self._k),
                                      algorithm="auto",
                                      init="k-means++",
                                      n_jobs=self._parallel_executions)
        self._kmeans.fit(self._embeddings)

    def _map_embeddings_to_clusters(self):
        self._clusters = {k: [] for k in range(int(self._k))}

        for i, word in enumerate(self._model.docvecs.doctags):
            self._clusters[self._kmeans.labels_[i]].append(word)

        return self._clusters

    def name(self):
        return f"k-means-{self._k}"
