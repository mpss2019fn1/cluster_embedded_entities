from abc import abstractmethod, ABC
from pathlib import Path
from typing import Dict, List

from gensim.models import KeyedVectors

from util.utils import measure


class AbstractClusterBuilder(ABC):

    def __init__(self, input_model_path: Path, parallel_executions: int):
        self._input_model_path: Path = input_model_path
        self._parallel_executions: int = parallel_executions
        self._clusters: Dict[int, List[str]] = {}

    def build_clusters(self) -> None:
        measure(self._load_model, "loading model")
        measure(self._train_specific_clusters, "clustering")
        measure(self._map_embeddings_to_clusters, "mapping entities to clusters")

    def _load_model(self) -> None:
        self._model = KeyedVectors.load_word2vec_format(str(self._input_model_path.absolute()))
        self._embeddings = self._model.vectors

    @abstractmethod
    def _train_specific_clusters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _map_embeddings_to_clusters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def clusters(self) -> Dict[int, List[str]]:
        return self._clusters
