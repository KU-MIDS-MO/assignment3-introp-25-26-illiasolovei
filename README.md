[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/efvwT8DV)
# Introduction to Programming
# Winter semester 2025/26 — Assignment 3

#Note:
All concepts required to complete these exercises have been covered in the lectures and examples.
You should be able to solve each problem using only what has been taught so far:
basic Python, functions, and the NumPy material (arrays, shapes, slicing, ...).

# Tasks (each has a starter.py and a pytest file):

1) clean_and_scale_scores(scores, min_score, max_score)
   We have exam scores stored in a NumPy array (1D or 2D).
   - First, replace all values smaler than min_score by min_score,
     and all values larger than max_score by max_score.
    
   - Then linearly scale all values to the range [0, 1] using:
       scaled = (value - min_score) / (max_score - min_score)
   Return a new NumPy array of floats with the same shape as scores

2) count_values_in_bins(data, bin_edges)
   We want to count how many values fall into each numeric bin.
   - data is a 1-D NumPy array of numbers.
   - bin_edges is a 1-D NumPy array of length B+1, strictly increasing.
   These edges define B bins:
      Bin 0: [bin_edges[0], bin_edges[1])
      Bin 1: [bin_edges[1], bin_edges[2])
      ...
      Bin B-2: [bin_edges[B-2], bin_edges[B-1])
      Bin B-1: [bin_edges[B-1], bin_edges[B]]   (last bin is inclusive on the right)
   Values outside [bin_edges[0], bin_edges[-1]] are ignored.
   Return a 1-D NumPy array of length B with the counts per bin.

3) moving_average(signal, window_size)
   We want to smooth a 1-D NumPy array using a centered moving average.
   - signal is a 1-D NumPy array of numbers
   - window_size is a positive odd integer (1, 3, 5,...).
   Let k = (window_size - 1) // 2
   For each index i, consider the indices from max(0, i-k) to min(n-1, i+k),
   where n is the length of signal, and take the average of those values.
   Return a new 1-D NumPy array of floats with the same length as signal.

