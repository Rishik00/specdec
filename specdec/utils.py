import time
import matplotlib.pyplot as plt

class Timer:

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self):
        self.end = time.per_counter()
        self.time_elapsed = self.end_time - self.start_time


def plot_times():
    pass
