#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import numpy as np
from math import sqrt

NdArray = np.ndarray

def myHoughTransform(image: NdArray, rhoRes, thetaRes) -> Tuple[NdArray, NdArray, NdArray]:
	image_width, image_height = image.shape
	diagonal_length = sqrt(image_width**2 + image_height**2)
	M = diagonal_length + 1

	rho_bound = int(np.ceil(M / rhoRes))
	theta_bound = int(np.ceil(2 * np.pi / thetaRes))

	accumulator = np.zeros((rho_bound, theta_bound))

	for row in range(image_width):
		for column in range(image_height):
			if image[row, column] > 0:
				for theta in range(theta_bound):
					_theta = theta * thetaRes
					rho = column * np.cos(_theta) + row * np.sin(_theta)
					
					if rho >= 0:
						rho_index = int(rho // rhoRes)
						accumulator[rho_index, theta] += 1

	rho_scale = np.arange(0, M, rhoRes)
	theta_scale = np.arange(0, 2 * np.pi, thetaRes)

	# accumulator = np.zeros((len(rho_scale), len(theta_scale)))
	
	# for row in range(image_width):
	# 	for column in range(image_height):
	# 		if image[row, column] > 0:
	# 			for theta in theta_scale:
	# 				rho = column * np.cos(theta) + row * np.sin(theta)
	# 				rho_index = np.argmin(np.abs(rho_scale - rho))
	# 				theta_index = np.argmin(np.abs(theta_scale - theta))
	# 				accumulator[rho_index, theta_index] += 1

	return accumulator, rho_scale, theta_scale

def test_hough_transform() -> None:
	pass

if __name__ == "__main__":
	test_hough_transform()
