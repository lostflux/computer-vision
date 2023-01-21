#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Type
import numpy as np
from math import ( ceil, inf )
from scipy import signal    # For signal.gaussian function

NdArray = np.ndarray

from myImageFilter import myImageFilter

def myEdgeFilter(img0: NdArray, sigma: int | float) -> None:
	
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

	thetas = np.arctan2(image_gradients["y"], image_gradients["x"]) % np.pi
	thetas = normalize(np.rad2deg(thetas))

	print(f"thetas:\n{thetas}")

	for row in range(thetas.shape[0]):
		for column in range(thetas.shape[1]):
			gratitude_direction = thetas[row, column]
			image_gradients["xy"] = non_max_suppression(image_gradients["xy"], row, column, gratitude_direction)

	print(f"""
		image: {img0}
		smooth_image: {smooth_image}
		image_gradients["x"]: {image_gradients["x"]}
		image_gradients["y"]: {image_gradients["y"]}
		image_gradients["xy"]: {image_gradients["xy"]}
	""")


def normalize(theta: NdArray) -> NdArray:
	"""
		Normalize the angles in the array
		to the nearest 45-degree angle.
	"""
	norm = lambda x: np.round(x / 45) * 45
	return np.vectorize(norm)(theta)

def non_max_suppression(image: NdArray, row: int, column: int, theta: float) -> NdArray:
	"""
		Perform non-maximum suppression on the image.
	"""

	def in_range(direction="row") -> bool:
		if direction == "row":
			return row >= 0 and row < image.shape[0]
		else:
			return column >= 0 and column < image.shape[1]

	before, after = -inf, -inf
	current = image[row, column]

	if theta == 0: # E/W
		if column >= 1:
			before = image[row, column-1]
		if column < image.shape[1]-1: 
			after = image[row, column+1]

	elif theta == 45: # NW/SE
		if row >= 1 and column >= 1:
			before = image[row-1, column-1]
		if row < image.shape[0]-1 and column < image.shape[1]-1:
			after = image[row+1, column+1]

	elif theta == 90: # N/S
		if row >= 1:
			before = image[row-1, column]
		if row < image.shape[0]-1:
			after = image[row+1, column]

	elif theta == 135: # NE/SW
		if row >= 1 and column < image.shape[1]-1:
			before = image[row-1, column+1]
		if row < image.shape[0]-1 and column >= 1:
			after = image[row+1, column-1]
		
	else: raise ValueError(f"Invalid theta value: {theta}")
				
	highest = max(before, current, after)
	if current != highest: image[row, column] = 0
	print(f"""
		before: {before}
		current: {current}
		after: {after}
		image[{row}, {column}]: {image[row, column]}
	""")

	return image



def test_edge_filter() -> None:
	img = np.array([[1,1,1],[1,1,1],[1,1,1]])
	sigma = 2

	myEdgeFilter(img, sigma)

if __name__ == '__main__':
	test_edge_filter()	
