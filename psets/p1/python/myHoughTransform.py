#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import numpy as np
from math import sqrt

NdArray = np.ndarray

def myHoughTransform(image: NdArray, rhoRes, thetaRes) -> Tuple[NdArray, NdArray, NdArray]:
	"""
	
	"""

	image_width, image_height = image.shape
	M = sqrt(image_width**2 + image_height**2)

	rho_bound = int(np.ceil(M / rhoRes))
	theta_bound = int(np.ceil(2 * np.pi / thetaRes))

	accumulator = np.zeros((rho_bound, theta_bound))

	rho_scale = np.arange(0, M, rhoRes)
	theta_scale = np.arange(0, 2 * np.pi, thetaRes)

	for row in range(image_width):
		for column in range(image_height):
			if image[row, column] > 0:
				for theta_step, theta in enumerate(theta_scale):
					rho = column * np.cos(theta) + row * np.sin(theta)
					
					if rho >= 0:
						rho_index = int(rho // rhoRes)
						accumulator[rho_index, theta_step] += 1
						
	return accumulator, rho_scale, theta_scale

def test_hough_transform() -> None:
	pass

if __name__ == "__main__":
	test_hough_transform()
