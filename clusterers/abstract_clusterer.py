from abc import abstractmethod, ABC

from gensim.models import Doc2Vec

from util.utils import measure


class AbstractClusterer(ABC):

    def __init__(self, input_model_path, parallel_executions):
        self._input_model_path = input_model_path
        self._parallel_executions = parallel_executions
        self._clusters = None

    def build_clusters(self):
        measure(self._load_model, "loading model")
        measure(self._train_specific_clusters, "clustering")
        self._clusters = measure(self._map_embeddings_to_clusters, "mapping entities to clusters")

    def _load_model(self):
        self._model = Doc2Vec.load(self._input_model_path)
        self._embeddings = self._model.docvecs.vectors_docs

    @abstractmethod
    def _train_specific_clusters(self):
        raise NotImplementedError

    @abstractmethod
    def _map_embeddings_to_clusters(self):
        raise NotImplementedError

    @abstractmethod
    def name(self):
        raise NotImplementedError

    def clusters(self):
        return self._clusters
