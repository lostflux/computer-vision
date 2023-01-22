#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Type
import numpy as np
import scipy

NdArray = np.ndarray

def myImageFilter(image: NdArray, h: NdArray) -> NdArray:
	"""
		This is a function that convolves an image with a filter

		Parameters
		----------
			img0: NdArray
				The image to be filtered.
			h: NdArray
				The convolution filter.

		Returns
		-------
			filtered_image: NdArray
				The image obtained after applying the filter on the image.
	"""
	
	image_height, image_width = image.shape
	filter_dims = h.shape

	#? pad zeros at the edges of the image as appropriate.
	pad_height, pad_width = map(lambda x: x//2, filter_dims)
	padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)))

	#? compute filtered image...
	filtered_image = np.zeros((image_height, image_width))
	for row in range(image_height):
		for column in range(image_width):
			window_start = (row, column)
			window_end = (row + filter_dims[0], column + filter_dims[1])
			window = padded_image[window_start[0]:window_end[0], window_start[1]:window_end[1]]
			filtered_image[row, column] = np.sum(window * h)

	return filtered_image



#####################? TESTS #####################

def test_image_filter() -> None:
	img = np.array([[1,1,1],[1,1,1],[1,1,1]])
	h = np.array([[0,0,0],[0,1,0],[0,0,0]])
	h2 = np.array([[1,1,1],[1,1,1],[1,1,1]])
	h3 = np.array([[1,1],[1,1]])

	for filter in [h, h2, h3]:
		
		# filtered = myImageFilter(img, i)

		print(f"""
			Image: \t\t{[ list(i) for i in img ]}
			Filter: \t{[ list(j) for j in filter] }
			Result: \t{[ list(j) for j in myImageFilter(img, filter) ]}
			NumPy Result: \t{[ list(j) for j in scipy.signal.convolve2d(img, filter, mode="same") ]}
		""")

	# assert np.allclose(filtered, np.array([[2.0, 2.0, -2.0], [5.0, 2.0, -5.0], [8.0, 2.0, -8.0]]))

	# generate 1000 x 1000 numpy array containing random numbers between 0 and 1.
	img = np.random.rand(1000, 1000)

	# create a 3x3 filter with all 1s.
	h = np.ones((3,3))
	print(f"h: {h}")
	print(f"""
		Result: \n{myImageFilter(img, h)}
		Numpy Result: \n{scipy.signal.convolve2d(img, h, mode="same")}
	""")

if __name__ == "__main__":
	test_image_filter()
	
