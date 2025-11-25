# 3) moving_average(signal, window_size)
#    We want to smooth a 1-D NumPy array using a centered moving average.
#    - signal is a 1-D NumPy array of numbers
#    - window_size is a positive odd integer (1, 3, 5,...).
#    Let k = (window_size - 1) // 2
#    For each index i, consider the indices from max(0, i-k) to min(n-1, i+k),
#    where n is the length of signal, and take the average of those values.
#    Return a new 1-D NumPy array of floats with the same length as signal.

import numpy as np


def moving_average(signal, window_size):
    n = signal.shape[0]
    res = np.zeros(signal.shape, dtype=float)
    k = (window_size - 1) // 2

    for i in range(n):
        start = i - k
        if start < 0:
            start = 0
        end = i + k
        if end > n - 1:
            end = n - 1

        total = 0.0
        count = 0
        for j in range(start, end + 1):
            total += signal[j]
            count += 1
        res[i] = total / count

    return res
