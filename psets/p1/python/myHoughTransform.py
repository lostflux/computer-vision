#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import numpy as np

NdArray = np.ndarray

def myHoughTransform(image, rhoRes, thetaRes) -> Tuple[NdArray, NdArray, NdArray]:
	image_width, image_height = image.shape
	diagonal_length = np.sqrt(image_width**2 + image_height**2)
	M = diagonal_length + 1
	rho_scale = np.arange(0, M, rhoRes)
	theta_scale = np.arange(0, 2 * np.pi, thetaRes)

	accumulator = np.zeros((len(rho_scale), len(theta_scale)))
	
	for row in range(image_width):
		for column in range(image_height):
			if image[row, column] > 0:
				for theta_index, theta in enumerate(theta_scale):
					rho = column * np.cos(theta) + row * np.sin(theta)
					rho_index = np.argmin(np.abs(rho_scale - rho))
					accumulator[rho_index, theta_index] += 1

	return accumulator, rho_scale, theta_scale

def test_hough_transform() -> None:
	pass

if __name__ == "__main__":
	test_hough_transform()
