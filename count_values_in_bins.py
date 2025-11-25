# 2) count_values_in_bins(data, bin_edges)
#    We want to count how many values fall into each numeric bin.
#    - data is a 1-D NumPy array of numbers.
#    - bin_edges is a 1-D NumPy array of length B+1, strictly increasing.
#    These edges define B bins:
#       Bin 0: [bin_edges[0], bin_edges[1])
#       Bin 1: [bin_edges[1], bin_edges[2])
#       ...
#       Bin B-2: [bin_edges[B-2], bin_edges[B-1])
#       Bin B-1: [bin_edges[B-1], bin_edges[B]]   (last bin is inclusive on the right)
#    Values outside [bin_edges[0], bin_edges[-1]] are ignored.
#    Return a 1-D NumPy array of length B with the counts per bin.

import numpy as np


def count_values_in_bins(data, bin_edges):
    flattened = data.reshape(data.size)
    B = bin_edges.shape[0] - 1
    count = np.zeros(B, dtype=int)

    for b in range(B):
        left = bin_edges[b]
        right = bin_edges[b + 1]
        for i in range(flattened.size):
            x = flattened[i]
            if b < B - 1:
                if (x >= left) and (x < right):
                    count[b] += 1
            else:
                if (x >= left) and (x <= right):
                    count[b] += 1
    return count
