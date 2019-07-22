import sys
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, TextIO, Iterable

from clustering.abstract_cluster_builder import AbstractClusterBuilder


class AbstractClusterWriter(ABC):

    def write(self, cluster_builder: AbstractClusterBuilder, sink: Optional[Path] = None) -> None:
        if sink:
            self._write_to_file(cluster_builder, sink)
        else:
            self._write_to_output(cluster_builder, sys.stdout)

    def _write_to_file(self, cluster_builder: AbstractClusterBuilder, sink: Path) -> None:
        with open(Path(sink.absolute(), f"{cluster_builder.name()}.{self._file_extension()}"), "w+") as output:
            self._write_to_output(cluster_builder, output)

    def _write_to_output(self, cluster_builder: AbstractClusterBuilder, output: TextIO):
        print(*self._generate_output(cluster_builder), sep="\n", end="\n", file=output)

    @abstractmethod
    def _generate_output(self, cluster_builder: AbstractClusterBuilder) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def _file_extension(self):
        raise NotImplementedError
