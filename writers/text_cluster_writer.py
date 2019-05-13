from writers.abstract_cluster_writer import AbstractClusterWriter


class TextClusterWriter(AbstractClusterWriter):

    def _generate_output(self, clusterer, sink=None):
        for i in clusterer.clusters().keys():
            yield f"[[CLUSTER {i}]] with {len(clusterer.clusters()[i])} entities"
            yield from clusterer.clusters()[i]


