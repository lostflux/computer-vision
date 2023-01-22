#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from typing import Tuple, List

NdArray = np.ndarray

def myHoughLines(H: NdArray, num_lines: int) -> Tuple[List, List]:
	"""Find the `n` strongest lines in the Hough accumulator array.

	Parameters
	----------
	H : NdArray
		The Hough accumulator array.
	num_lines : int
		The number of lines to return.
	"""
	# first, pick out the n strongest lines

	print(f"H: \n{H}")
	print(f"H.shape: {H.shape}")
	height, width = H.shape
	# raise NotImplementedError

	# rhos, thetas = [], []
	threshold = 1  # The threshold for adjacent pixels

	for row in range(height):
		for col in range(width):
			for adjacent_row in range(-threshold, threshold+1):
				row_index = row + adjacent_row
				if 0 <= row_index < H.shape[0]:
					for adjacent_column in range(-threshold, threshold+1):
						col_index = col + adjacent_column
						if 0 <= col_index < H.shape[1]:
							if H[row_index, col_index] > H[row, col]:
								H[row, col] = 0



	rhos, thetas = [], []
	ind = np.argpartition(H.ravel(), H.size - num_lines)[-num_lines:]
	ind = np.column_stack(np.unravel_index(ind, H.shape))
	print(ind)
	
	for line in range(num_lines):
		rhos.append(int(ind[line][0]))
		thetas.append(int(ind[line][1]))

	return rhos, thetas
