#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Amittai Siavava"
__github__ = "@siavava"

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

    diagonal = np.hypot(image.shape[0], image.shape[1])
    rho_scale = np.arange(0, diagonal, rhoRes)
    theta_scale = np.arange(0, 2*np.pi, thetaRes)

    img_hough = np.zeros(shape=(len(rho_scale), len(theta_scale)))
    edge_points = np.transpose(np.nonzero(image))
    
    # print(f"{rho_scale =  }")
    # print(f"{theta_scale =  }")

    #? for each edge point, calculate the rhos and thetas
    #? and accumulate votes.
    for y, x in edge_points:
        
        # calculate rhos corresponding to the thetas in thetaScale
        rhos = x * np.cos(theta_scale) + y * np.sin(theta_scale)
        theta_index = np.where(rhos >= 0)
        rhos = rhos[rhos >= 0]
        
        # NOTE: used numpy reference for digitize
        # https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
        rho_index = np.digitize(rhos, rho_scale)
        img_hough[rho_index, theta_index] += 1
    
    return img_hough, rho_scale, theta_scale

def test_hough_transform() -> None:
	return NotImplemented

if __name__ == "__main__":
	test_hough_transform()
