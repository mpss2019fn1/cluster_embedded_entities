from sklearn import cluster

from clusterers.abstract_clusterer import AbstractClusterer


class DBScanClusterer(AbstractClusterer):

    def __init__(self, input_model_path, parallel_executions):
        AbstractClusterer.__init__(self, input_model_path, parallel_executions)

    def _train_specific_clusters(self):
        raise NotImplementedError

    def _map_embeddings_to_clusters(self):
        raise NotImplementedError

    def name(self):
        return f"DBSCAN"
