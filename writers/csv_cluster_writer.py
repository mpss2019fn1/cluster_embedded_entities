from writers.abstract_cluster_writer import AbstractClusterWriter


class CSVClusterWriter(AbstractClusterWriter):

    def _generate_output(self, clusterer, sink=None):
        yield "cluster_id,entity"
        for i in range(len(clusterer.clusters())):
            yield from [f"{i},{value}" for value in clusterer.clusters()[i]]

