import logging
import time


def measure(function_to_execute, function_name):
    start_time = time.perf_counter()
    return_value = function_to_execute()
    end_time = time.perf_counter()

    logging.info(f"Execution of {function_name} took {end_time - start_time} seconds.")
    return return_value
