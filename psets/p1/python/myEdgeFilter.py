#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Amittai Siavava"
__github__ = "@siavava"

import numpy as np
from math import ( ceil, inf )
from scipy import signal
import cv2

NdArray = np.ndarray

from myImageFilter import myImageFilter

def myEdgeFilter(img0: NdArray, sigma):
    """
        This is a function that filters an image with a Gaussian filter and then finds the edges
        using the Sobel operator.
        
        Parameters
        ----------
        img0: NdArray
            The image to be filtered.
        sigma: float
            The standard deviation of the Gaussian filter.
        
        Returns
        -------
        image: NdArray
            The image obtained after applying the edge filter.
    """
    #? create 2D gaussian kernel
    gaussian = generate_gaussian(sigma)

    smoothed = myImageFilter(img0, gaussian)

    #? apply sobel filters in X and Y directions
    # NOTE: we apply them as two 1D convolutions to be more computationally efficient
    gradients_x = myImageFilter(myImageFilter(smoothed, np.array([[1], [2], [1]])), np.array([[1, 0, -1]]))
    gradients_y = myImageFilter(myImageFilter(smoothed, np.array([[1], [0], [-1]])),  np.array([[1, 2, 1]]))

    # NOTE: reference for arctan2 vs arctan:
    #   https://www.scaler.com/topics/numpy-arctan2/
    direction = np.rad2deg(np.arctan2(gradients_y, gradients_x))
    magnitude = np.hypot(gradients_x, gradients_y)

    # NOTE: normalize direction to be between 0 and 180
    direction[direction < 0] += 180

    #? non-maxima suppression in relevant directions
    image = nms_directed(magnitude, direction)

    #? mute image edges
    image[0] = image[:, 0] = image[:, -1] = image[-1, :] = 0
    
    return image

def generate_gaussian(sigma) -> NdArray:
    """
    Generates a gaussian filter of the specified size and standard deviation.
    """
    
    #? generate 1D gaussian kernel
    hsize = 2 * np.ceil(3 * sigma) + 1
    kernel = signal.gaussian(hsize, std=sigma)
    
    #? outer product to get 2D kernel
    h = np.outer(kernel, kernel)
    
    #? normalize
    h /= h.sum()
    
    return h


def nms_directed(magnitude: NdArray, direction: NdArray) -> NdArray:
    """Perform non-maxima suppression in the direction of the gradient."""

    #? directional kernels
    kernels = {
        0: np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),
        45: np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8),
        90: np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8),
        135: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    }
    
    dilations = { angle: cv2.dilate(magnitude, kernel) for angle, kernel in kernels.items() }
    
    diffs = { angle: np.nonzero(np.absolute(dilations[angle] - magnitude)) for angle in dilations.keys() }
    
    # suppress non-maxima in each relevant direction
    for angle in dilations:
        dilations[angle][diffs[angle]] = 0
        
    #? index angles from range [0, 180) to [0, 3]
    indices = index(direction)

    image = np.choose(indices, np.stack([dilations[0], dilations[45], dilations[90], dilations[135]]))
    
    return image


def index(gradient: NdArray) -> NdArray:
    """
    Index angles from range [0, 180) to [0, 3]
    
    Mappings
    --------
    - 0: 0 <= theta <22.5 UNION 157.5 <= theta < 180
    - 1: 22.5 <= theta < 67.5
    - 2: 67.5 <= theta < 112.5
    - 3: 112.5 <= theta < 157.5
    
    Parameters
    ----------
    gradient : NdArray
        The angles of gradients to be indexed.
    
    Returns
    -------
    NdArray
        The indices of the angles.
    
    """

    #? discretize the angles to 0, 45, 90, 135
    # indices = np.zeros(shape=gradient.shape, dtype=int)
    # indices = np.where( (22.5 <= gradient)    & (gradient < 67.5), indices, 1)
    # indices = np.where( (67.5 <= gradient)    & (gradient < 112.5), indices, 2)
    # indices = np.where( (112.5 <= gradient)   & (gradient < 157.5), indices, 3)
    
    m = 22.5
    
    indices = np.zeros(shape=gradient.shape, dtype=int)
    
    indices = np.where((gradient < (135 + m)) & (gradient >= m), indices, 0)          # change angles in range [0, 22.5) and [157.5, 180] to 0 
    indices = np.where((gradient < m) | (gradient >= (45 + m)), indices, 1)           # change angles in range [22.5, 67.5) to 1
    indices = np.where((gradient < (90 - m)) | (gradient >= (90 + m)), indices, 2)    # change angles in range [67.5, 112.5) to 2
    indices = np.where((gradient < (135 - m)) | (gradient >= (135 + m)), indices, 3)  # change angles in range [112.5, 157.5) to 3
    return indices


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
