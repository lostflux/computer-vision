#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Type
import numpy as np

NpArray = np.ndarray

def myImageFilter(img0: NpArray, h: NpArray) -> NpArray:
	"""This is a function that convolves an image with a filter

	Parameters
	----------
		img0: Type(np.array)
			The image to be filtered
		h (_type_): _description_
	"""
	
	image_height = img0.shape[0]
	image_width = img0.shape[1]
	filter_dims = h.shape

	# pad zeros at the edges of the image.
	pad_height, pad_width = map(lambda x: x//2, filter_dims)
	padded_image = np.pad(img0, ((pad_height, pad_height), (pad_width, pad_width)))

	# create the filtered image and compute values.
	filtered_image = np.zeros((image_height, image_width))

	for row in range(image_height):
		for column in range(image_width):
			window_start = (row, column)
			window_end = (row + filter_dims[0], column + filter_dims[1])
			window = padded_image[window_start[0]:window_end[0], window_start[1]:window_end[1]]
			filtered_image[row, column] = np.sum(window * h)

	return filtered_image

def test_filter() -> None:
	img = np.array([[1,2,3],[4,5,6],[7,8,9]])
	h = np.array([[1,1],[1,1]])
	filtered = myImageFilter(img, h)

	print(f"""
		Image: \t\t{[ list(i) for i in img ]}
		Filter: \t{[ list(i) for i in h] }
		Expected: \t[[1,3,5],[5,12,16],[11,24,28]]
		Actual: \t{[ list(i) for i in filtered ]}
	""")

	assert np.allclose(filtered, np.array([[1,3,5],[5,12,16],[11,24,28]]))

if __name__ == "__main__":
	test_filter()
	
