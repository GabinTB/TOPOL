import time
from functools import wraps
from typing import Union, Callable, Any

# Define the rate limit decorator
def rate_limit(calls_per_minute: Union[int, float, Callable]) -> Callable:
    """
    A decorator to limit function calls to a specific number per minute.
    :param calls_per_minute: Maximum number of function calls per minute
    """
    if isinstance(calls_per_minute, Callable):
        calls_per_minute = calls_per_minute()
        if not isinstance(calls_per_minute, int):
            raise ValueError("calls_per_minute must return an int.")
    elif isinstance(calls_per_minute, float):
        calls_per_minute = int(calls_per_minute)
    elif not isinstance(calls_per_minute, int):
        raise ValueError("calls_per_minute must be an int, a float or a callable that returns an int")
    interval = 60 / calls_per_minute  # Calculate the time interval between calls
    last_call_time = [0]  # Store last call time (mutable to be used in inner function)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed_time = current_time - last_call_time[0]

            # If the time between the last call and current call is less than the required interval
            if elapsed_time < interval:
                time_to_wait = interval - elapsed_time
                time.sleep(time_to_wait)  # Sleep for the remaining time

            last_call_time[0] = time.time()  # Update last call time
            return func(*args, **kwargs)

        return wrapper

    return decorator