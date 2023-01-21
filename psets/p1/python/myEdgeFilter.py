#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Type
import numpy as np
from math import ( ceil, inf )
from scipy import signal    # For signal.gaussian function

NdArray = np.ndarray

from myImageFilter import myImageFilter

def myEdgeFilter(img0: NdArray, sigma: int | float) -> NdArray:
	
	hsize = 2 * ceil(3 * sigma) + 1

	# NOTE: signal.gaussian is migrated to signal.windows.gaussian in Python 3.11
	gaussian_filter = signal.windows.gaussian(hsize, sigma).reshape(hsize,1)

	smooth_image = myImageFilter(img0, gaussian_filter)

	sobel_filter = {
		"x": np.array([[1,0,-1],[2,0,-2],[1,0,-1]]),
		"y": np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	}

	image_gradients = {
		"x": myImageFilter(smooth_image, sobel_filter["x"]),
		"y": myImageFilter(smooth_image, sobel_filter["y"]),
	}

	image_gradients["xy"] = np.sqrt(image_gradients["x"]**2 + image_gradients["y"]**2)

	gradient_direction = np.arctan2(image_gradients["y"], image_gradients["x"]) % np.pi
	gradient_direction = normalize(np.rad2deg(gradient_direction))

	# print(f"thetas:\n{gradient_direction}")

	for row in range(gradient_direction.shape[0]):
		for column in range(gradient_direction.shape[1]):
			gratitude_direction = gradient_direction[row, column]
			image_gradients["xy"] = non_max_suppression(image_gradients["xy"], row, column, gratitude_direction)

	# print(f"""
	# 	image: {img0}
	# 	smooth_image: {smooth_image}
	# 	image_gradients["x"]: {image_gradients["x"]}
	# 	image_gradients["y"]: {image_gradients["y"]}
	# 	image_gradients["xy"]: {image_gradients["xy"]}
	# """)
	return image_gradients["xy"]


def normalize(theta: NdArray) -> NdArray:
	"""
		Normalize the angles in the array
		to the nearest 45-degree angle.
	"""
	norm = lambda x: (np.round(x / 45) * 45) % 180
	return np.vectorize(norm)(theta)

def non_max_suppression(image: NdArray, row: int, column: int, angle: float) -> NdArray:
	"""
		Perform non-maximum suppression on the image.

		Parameters
		----------
		image : NdArray
			The image to perform non-maximum suppression on.
		row : int
			The row of the pixel to perform non-maximum suppression on.
		column : int
			The column of the pixel to perform non-maximum suppression on.
		angle : float
			The angle of the gradient at the pixel.
	"""

	before, after = -inf, -inf
	current = image[row, column]

	if angle == 0:
		"""
			. . .
			x c x
			. . .
		"""
		if column >= 1:
			before = image[row, column-1]
		if column < image.shape[1]-1: 
			after = image[row, column+1]

	elif angle == 45:
		"""
			x . .
			. c .
			. . x
		"""
		if row >= 1 and column >= 1:
			before = image[row-1, column-1]
		if row < image.shape[0]-1 and column < image.shape[1]-1:
			after = image[row+1, column+1]

	elif angle == 90:
		"""
			. x .
			. c .
			. x .
		"""
		if row >= 1:
			before = image[row-1, column]
		if row < image.shape[0]-1:
			after = image[row+1, column]

	elif angle == 135:
		"""
			. . x
			. c .
			x . .
		"""
		if row >= 1 and column < image.shape[1]-1:
			before = image[row-1, column+1]
		if row < image.shape[0]-1 and column >= 1:
			after = image[row+1, column-1]
		
	else: raise ValueError(f"Invalid angle: {angle}")
				
	highest = max(before, current, after)
	if current != highest: image[row, column] = 0
	# print(f"""
	# 	before: {before}
	# 	current: {current}
	# 	after: {after}
	# 	image[{row}, {column}]: {image[row, column]}
	# """)

	return image



def test_edge_filter() -> None:
	img = np.array([[1,1,1],[1,1,1],[1,1,1]])
	sigma = 2

	myEdgeFilter(img, sigma)

if __name__ == '__main__':
	test_edge_filter()	
