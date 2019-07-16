import logging
import time
from typing import Callable, Any, TypeVar

T = TypeVar("T")


def measure(function_to_execute: Callable[[], T], function_name) -> T:
    start_time: float = time.perf_counter()
    return_value: Any = function_to_execute()
    end_time: float = time.perf_counter()

    logging.info(f"Execution of {function_name} took {end_time - start_time} seconds")
    return return_value
