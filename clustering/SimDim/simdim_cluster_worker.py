import logging
import statistics
import time
from math import ceil
from typing import List, Tuple, Optional, Dict

from gensim.models import KeyedVectors


class SimDimClusterWorker:

    def __init__(self, model: KeyedVectors):
        self._model: KeyedVectors = model
        self._minimum_cluster_size: int = max(10, len(self._model.vectors) // (self._model.vector_size * 10))

        # state for cluster extraction
        self._dimension: int = 0
        self._sorted_values: List[float] = []
        self._sorted_labels: List[str] = []
        self._tolerance: float = 0.0
        self._biggest_cluster: Tuple[int, int] = (0, 0)
        self._current_cluster: Tuple[int, int] = (0, 0)
        self._current_start_index: int = 0
        self._current_end_index: int = 0

    def __call__(self, dimension: int):
        return self.extract_cluster(dimension)

    def extract_cluster(self, dimension: int) -> Optional[Dict[int, List[str]]]:
        self._dimension = dimension

        logging.info(f"[DIMENSION-{self._dimension}] begin")

        vector_values: List[float] = [vector[self._dimension].item() for vector in self._model.vectors]
        vector_labels: List[str] = list(self._model.vocab.keys())

        sorted_tuples: List[Tuple[float, str]] = sorted(zip(vector_values, vector_labels), key=lambda x: x[0])

        self._sorted_values = [x[0] for x in sorted_tuples]
        self._sorted_labels = [x[1] for x in sorted_tuples]
        self._tolerance = statistics.mean(self._sorted_values)
        self._create_biggest_cluster()

        logging.info(f"[DIMENSION-{self._dimension}] done")

        if self._len(self._biggest_cluster) < self._minimum_cluster_size:
            return None

        return {self._dimension: [self._sorted_labels[label_index]
                                  for label_index in range(self._biggest_cluster[0], self._biggest_cluster[1])]}

    def _create_biggest_cluster(self) -> None:
        self._biggest_cluster = (0, 0)
        self._current_start_index = 0

        execution_times: List[float] = []
        log_interval: int = len(self._sorted_values) // 10

        while self._current_start_index < len(self._sorted_values) - self._min_len():
            start_time: float = time.perf_counter()
            self._create_current_cluster()

            if self._len(self._current_cluster) > self._len(self._biggest_cluster):
                self._biggest_cluster = self._current_cluster

            execution_times.append(time.perf_counter() - start_time)

            if len(execution_times) % log_interval == 0:
                progression: float = self._current_start_index / len(self._sorted_values)
                total_time: float = sum(execution_times)
                logging.info(f"[DIMENSION-{self._dimension}]\t{'%06.2f%%' % (progression * 100.0)} "
                             f"avg: {'%06.4fs' % (total_time / len(execution_times))}; "
                             f"tot: {'%06.4fs' % total_time}; "
                             f"<<_biggest_cluster.len>>: {self._len(self._biggest_cluster)}")

                execution_times.clear()

    def _create_current_cluster(self) -> None:
        self._current_cluster = (0, 0)
        self._current_end_index = min(len(self._sorted_values) - 1,
                                      self._current_start_index + self._min_len())

        if self._end() - self._start() > self._tolerance:
            self._current_start_index = min(len(self._sorted_values) - 1,
                                            self._current_start_index + 1)
            return

        while self._current_end_index < len(self._sorted_values) - 1:
            last_confirmed_start_index: int = self._current_end_index
            self._current_end_index = min(len(self._sorted_values) - 1,
                                          self._current_end_index + self._min_len())

            difference: float = self._end() - self._start()

            if difference < self._tolerance:
                continue

            if difference == self._tolerance:
                while difference == self._tolerance and self._current_end_index < len(self._sorted_values) - 1:
                    self._current_end_index += min(len(self._sorted_values) - 1,
                                                   self._current_end_index + 1)
                    difference = self._end() - self._start()

            else:  # difference > self._tolerance
                self._current_end_index = self._find_last_conditional_index_in_range(last_confirmed_start_index)
            break

        self._current_cluster = (self._current_start_index, self._current_end_index)
        self._find_next_start_index()

    def _find_next_start_index(self) -> None:
        if self._current_end_index - self._current_start_index < 2:
            self._current_start_index = self._current_end_index
            return

        self._current_start_index = self._find_first_conditional_index_in_range(self._current_start_index + 1)

    def _find_first_conditional_index_in_range(self, start: int) -> int:
        end: int = self._current_end_index
        pivot: int = start

        while end - start > 1:
            pivot = start + ceil((end - start) / 2)
            difference: float = self._sorted_values[self._current_end_index] - self._sorted_values[pivot]

            if difference > self._tolerance:
                start = pivot
                continue

            if difference == self._tolerance:
                while difference == self._tolerance:
                    pivot -= 1
                    difference = self._sorted_values[self._current_end_index] - self._sorted_values[pivot]
                return pivot + 1

            # Difference < self.tolerance
            end = pivot

        return pivot

    def _find_last_conditional_index_in_range(self, start: int) -> int:
        end: int = self._current_end_index
        pivot: int = start

        while end - start > 1:
            pivot = start + ceil((end - start) / 2)
            difference: float = self._sorted_values[pivot] - self._sorted_values[self._current_start_index]

            if difference > self._tolerance:
                end = pivot
                continue

            if difference == self._tolerance:
                while difference == self._tolerance and pivot < self._current_end_index:
                    pivot += 1
                    difference = self._sorted_values[pivot] - self._sorted_values[self._current_start_index]
                return pivot

            # Difference < self.tolerance
            start = pivot

        return pivot

    def _start(self) -> float:
        return self._sorted_values[self._current_start_index]

    def _end(self) -> float:
        return self._sorted_values[self._current_end_index]

    @staticmethod
    def _len(cluster: Tuple[int, int]) -> int:
        return cluster[1] - cluster[0]

    def _min_len(self) -> int:
        return max(self._minimum_cluster_size, self._len(self._biggest_cluster))
