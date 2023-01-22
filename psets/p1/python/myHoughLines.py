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

	rho_count, theta_count = H.shape
	threshold = 1  							#? 1 = 3x3, 2 = 5x5, 3 = 7x7, etc.

	for rho in range(rho_count):
		for theta in range(theta_count):
			for adjacent_rho in range(-threshold, threshold+1):
				rho_index = rho + adjacent_rho
				if 0 <= rho_index < H.shape[0]:
					for adjacent_theta in range(-threshold, threshold+1):
						theta_index = theta + adjacent_theta
						if 0 <= theta_index < H.shape[1]:
							if H[rho_index, theta_index] > H[rho, theta]:
								H[rho, theta] = 0

	rhos, thetas = [], []
	lines = np.argpartition(H.ravel(), H.size - num_lines)[-num_lines:]
	lines = np.column_stack(np.unravel_index(lines, H.shape))
	
	for line in range(num_lines):
		rhos.append(int(lines[line][0]))
		thetas.append(int(lines[line][1]))

	return rhos, thetas
