import sys
from abc import abstractmethod, ABC
from pathlib import Path


class AbstractClusterWriter(ABC):

    def write(self, clusterer, sink=None):
        if sink:
            self._write_to_file(clusterer, sink)
        else:
            self._write_to_output(clusterer, sys.stdout)

    def _write_to_file(self, clusterer, sink):
        output_file_location = Path(sink)
        with open(Path(output_file_location, f"{clusterer.name()}.txt"), "w+") as output:
            self._write_to_output(clusterer, output)

    def _write_to_output(self, clusterer, output):
        print(*self._generate_output(clusterer), sep="\n", end="\n", file=output)

    @abstractmethod
    def _generate_output(self, clusterer):
        raise NotImplementedError
