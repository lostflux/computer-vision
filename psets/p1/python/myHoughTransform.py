#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import numpy as np
from math import sqrt

NdArray = np.ndarray

def myHoughTransform(image: NdArray, rhoRes, thetaRes) -> Tuple[NdArray, NdArray, NdArray]:
	"""
		Computes the Hough Transform of the image.

		Parameters
		----------
		image : NdArray
			The image to compute the Hough Transform of.
		rhoRes : float
			The resolution of the rho axis.
		thetaRes : float
			The resolution of the theta axis.

		Returns
		-------
		Tuple[NdArray, NdArray, NdArray]
			The accumulator array, the rho axis, and the theta axis.
	"""

	image_width, image_height = image.shape
	M = sqrt(image_width**2 + image_height**2)

	rho_scale = np.arange(0, M, rhoRes)
	theta_scale = np.arange(0, 2 * np.pi, thetaRes)

	#? accumulate votes
	accumulator = np.zeros((len(rho_scale), len(theta_scale)))
	for row in range(image_width):
		for column in range(image_height):
			if image[row, column] > 0:
				for theta_step, theta in enumerate(theta_scale):
					rho = column * np.cos(theta) + row * np.sin(theta)
					
					if rho >= 0:
						rho_step = int(rho // rhoRes)
						accumulator[rho_step, theta_step] += 1
						
	return accumulator, rho_scale, theta_scale

def test_hough_transform() -> None:
	raise NotImplementedError

if __name__ == "__main__":
	test_hough_transform()
