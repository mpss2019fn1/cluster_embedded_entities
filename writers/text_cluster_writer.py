from collections import Iterable

from clustering.abstract_cluster_builder import AbstractClusterBuilder
from writers.abstract_cluster_writer import AbstractClusterWriter


class TextClusterWriter(AbstractClusterWriter):

    def _generate_output(self, cluster_builder: AbstractClusterBuilder) -> Iterable[str]:
        for i in cluster_builder.clusters().keys():
            yield f"[[CLUSTER {i}]] with {len(cluster_builder.clusters()[i])} entities"
            yield from cluster_builder.clusters()[i]


