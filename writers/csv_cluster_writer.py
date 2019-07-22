from typing import Iterable

from clustering.abstract_cluster_builder import AbstractClusterBuilder
from writers.abstract_cluster_writer import AbstractClusterWriter


class CSVClusterWriter(AbstractClusterWriter):

    def _generate_output(self, cluster_builder: AbstractClusterBuilder) -> Iterable[str]:
        yield "cluster_id,entity"
        for i in cluster_builder.clusters().keys():
            yield from [f"{i},{value}" for value in cluster_builder.clusters()[i]]


