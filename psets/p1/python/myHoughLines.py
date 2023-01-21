#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2  # For cv2.dilate function

from typing import Tuple, List

NdArray = np.ndarray

def myHoughLines(H: NdArray, nLines: int) -> Tuple[List, List]:
	"""This is a function that finds the nLines strongest lines in the Hough

	Args:
		H {int}: _description_
		nLines (_type_): _description_
	"""
	# YOUR CODE HERE
	rhos, thetas = [], []

	# Create a copy of H
	# H = np.copy(H)
	
	threshold = 2
	for line_number in range(nLines):
		line_candidate = np.argmax(H)
		line_location = np.unravel_index(line_candidate, H.shape)
		rhos.append(line_location[0])
		thetas.append(line_location[1])
		# H[line_location[0], line_location[1]] = 0

		for adjacent_row in range(-threshold, threshold+1):
			row_index = line_location[0] + adjacent_row
			if 0 <= row_index < H.shape[0]:
				for adjacent_column in range(-threshold, threshold+1):
					if adjacent_row == 0 and adjacent_column == 0:
						col_index = line_location[1] + adjacent_column
						if 0 <= col_index < H.shape[1]:
							# do nothing if the point is the line candidate
							H[row_index, col_index] = 0

	return rhos, thetas
