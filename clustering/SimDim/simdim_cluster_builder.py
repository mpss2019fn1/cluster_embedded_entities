from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Iterable, Dict

from clustering.SimDim.simdim_cluster_worker import SimDimClusterWorker
from clustering.abstract_cluster_builder import AbstractClusterBuilder


class SimDimClusterBuilder(AbstractClusterBuilder):

    def __init__(self, input_model_path: Path, parallel_executions: int):
        super(SimDimClusterBuilder, self).__init__(input_model_path, parallel_executions)

    def _train_specific_clusters(self) -> None:
        self._clusters: Dict[int, Iterable[str]] = {}

        dimensions: Iterable[int] = (dimension for dimension in range(self._embeddings.vector_size))
        with Pool(processes=self._parallel_executions) as pool:
            clusters: List[Dict[int, Iterable[str]]] = pool.map(SimDimClusterWorker(self._embeddings), dimensions)

        for cluster in clusters:
            if not cluster:
                continue

            dimension: int = list(cluster.keys())[0]
            self._clusters[dimension] = List[cluster[dimension]]

    def _map_embeddings_to_clusters(self) -> None:
        pass

    def name(self) -> str:
        return f"SIMDIM"
