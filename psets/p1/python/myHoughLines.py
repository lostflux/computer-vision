import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines: int):
	"""This is a function that finds the nLines strongest lines in the Hough

	Args:
		H {int}: _description_
		nLines (_type_): _description_
	"""
	# YOUR CODE HERE
	rhos, thetas = [], []

	# Create a copy of H
	array_copy = np.copy(H)
	
	threshold = 7
	for line in range(nLines):
		line_location = np.unravel_index(np.argmax(array_copy), array_copy.shape)
		rhos.append(line_location[0])
		thetas.append(line_location[1])
		array_copy[line_location[0], line_location[1]] = 0

		for adjacent_row in range(-threshold, threshold):
			row_index = line_location[0] + adjacent_row
			if 0 <= row_index < array_copy.shape[0]:
				for adjacent_column in range(-threshold, threshold):
					col_index = line_location[1] + adjacent_column
					if 0 <= col_index < array_copy.shape[1]:
						array_copy[row_index, col_index] = 0

	return rhos, thetas
