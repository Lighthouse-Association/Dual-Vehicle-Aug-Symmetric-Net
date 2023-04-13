import time
from datetime import timedelta


def log_func_latency(func, *args, **kwargs):
    start = time.monotonic()
    ret = func(*args, **kwargs)
    latency = timedelta(time.monotonic() - start)
    return latency, ret


class Timer:

    def __init__(self):
        self.start_time = time.monotonic()
        self.history = []

    def reset(self):
        self.start_time = time.monotonic()

    def update(self):
        return time.monotonic() - self.start_time
    
    def log_time(self):
        self.history.append(self.update())
        return self.history[-1]
    
    def __str__(self) -> str:
        return str(timedelta(seconds=self.update()))
