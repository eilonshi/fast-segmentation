import time
import datetime

from typing import Tuple


class TimeMeter(object):
    """
    A class for time measurement through iteration process
    """

    def __init__(self, max_iter):
        self.iter = 0
        self.max_iter = max_iter
        self.st = time.time()
        self.global_st = self.st
        self.curr = self.st

    def update(self):
        """
        updates the number of iterations passed

        Returns:
            None
        """
        self.iter += 1

    def get(self) -> Tuple[float, str]:
        """
        Calculates the time of the last iteration and the ETA (estimated time of arrival)

        Returns:
            a tuple of the last iteration time and a string of the ETA
        """
        self.curr = time.time()
        interval = self.curr - self.st
        global_interval = self.curr - self.global_st
        eta = int((self.max_iter - self.iter) * (global_interval / (self.iter + 1)))
        eta = str(datetime.timedelta(seconds=eta))
        self.st = self.curr

        return interval, eta


class AvgMeter(object):
    """
    A class for metric average measurement through time
    """

    def __init__(self, name):
        self.name = name
        self.seq = []
        self.global_seq = []

    def update(self, val):
        """
        Updates the lists of the metric - appends the new value

        Args:
            val: a new value for the metric

        Returns:
            None
        """
        self.seq.append(val)
        self.global_seq.append(val)

    def get(self) -> Tuple[float, float]:
        """
        Calculates the averages of the metric

        Returns:
            a tuple - (an average of the metric from the last call to this method, global average of the metric)
        """
        avg = sum(self.seq) / len(self.seq)
        global_avg = sum(self.global_seq) / len(self.global_seq)
        self.seq = []

        return avg, global_avg
