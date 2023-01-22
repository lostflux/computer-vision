#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from math import ( ceil, inf )
from scipy import signal

NdArray = np.ndarray

from myImageFilter import myImageFilter

SOBEL_FILTER = {
	"x": np.array([[1,0,-1],[2,0,-2],[1,0,-1]]),
	"y": np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
}


def myEdgeFilter(img0: NdArray, sigma: int | float) -> NdArray:
	"""
	
	"""

	gaussian_filter = generate_gaussian(sigma)
	smooth_image = normalized_image(myImageFilter(img0, gaussian_filter))
	image_gradients = {
		"x": myImageFilter(smooth_image, SOBEL_FILTER["x"]),
		"y": myImageFilter(smooth_image, SOBEL_FILTER["y"]),
	}

	image_gradients["xy"] = np.sqrt((image_gradients["x"]**2) + (image_gradients["y"]**2))
	grad = gradients(image_gradients["y"], image_gradients["x"])

	
	#! create copy of gradients to modify.
	output = image_gradients["xy"].copy()

	for row in range(grad.shape[0]):
		for column in range(grad.shape[1]):
			direction = grad[row, column]
			output = non_max_suppression(output, image_gradients["xy"], row, column, direction)

	return output

"""

							HELPER METHODS

"""

def normalized_image(image: NdArray) -> NdArray:
	"""
		Normalizes an array of pixel images
		to the range 0 <= x < 256.

		Parameters
		----------
		image : NdArray
			The array of image pixel values.

		Returns
		-------
		NdArray
			The normalized array of image pixel values.
	"""
	return image % 255 # np.vectorize(lambda x: x % 255)(image)



def normalized_gaussian(arr: NdArray) -> NdArray:
	"""
		Normalizes an array to sum up to `1` globally.
	"""
	return arr / np.sum(arr)



def gradients(y: NdArray, x: NdArray) -> NdArray:
	"""
		Normalize the angles in the array
		to the nearest 45-degree angle.
	"""
	grad = np.rad2deg(np.arctan2(y, x))
	return ((grad / 45).round() * 45) % 180



def generate_gaussian(sigma: int | float) -> NdArray:
	r"""
		Generates a square Gaussian filter of size 2 * ceil(3 * sigma) + 1
		NOTE: signal.gaussian is migrated to signal.windows.gaussian in Python 3.11

		See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html

		Parameters
		----------
		sigma : int | float
			The standard deviation of the Gaussian filter.
	"""

	hsize = 2 * ceil(3 * sigma) + 1
	gaussian_vector = signal.windows.gaussian(hsize, sigma)
	gaussian_filter = np.outer(gaussian_vector, gaussian_vector)
	return normalized_gaussian(gaussian_filter)



def non_max_suppression(output: NdArray, image: NdArray, row: int, column: int, angle: float) -> NdArray:
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

	current = image[row, column]
	highest = max(before, current, after)

	#! mute current index if it is not a local maximum.
	if current != highest:
		output[row, column] = 0

	return output


def test_edge_filter() -> None:
	img = np.array([[1,1,1],[1,1,1],[1,1,1]])
	
	for sigma in range(5):

		filtered = myEdgeFilter(img, sigma)
		print(f"""
			img: \n{img}
			sigma: {sigma}
			filtered: \n{filtered}
		""")

if __name__ == '__main__':
	test_edge_filter()	
