#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Amittai Siavava"
__github__ = "@siavava"

import numpy as np

from typing import Tuple, List
import cv2
NdArray = np.ndarray

def myHoughLines(H: NdArray, num_lines: int) -> Tuple[List, List]:
    """
    Find the `n` strongest lines in the Hough accumulator array.

    Parameters
    ----------
    H : NdArray
        The Hough accumulator array.
    num_lines : int
        The number of lines to return.
        
    Returns
    -------
    Tuple[List, List]
        The rhos and thetas of the `n` strongest lines.
    """

    img = H.copy()
    
    #? suppress non-maxima
    nms(img)

    #? return the top N lines
    return topN(img, num_lines)
    
def nms(img: NdArray) -> NdArray:
    """Performs non-maxima suppression on the image."""
    
    #? 3X3 kernel to look at all neighbors
    kernel = np.ones(shape=(3, 3))
    
    #? dilation replaces each pixel with the max of its 3x3 window
    dilated = cv2.dilate(img, kernel)
    
    #? find non-maxima pixels indices
    non_maxima = np.where(img != dilated)
    
    #? suppress
    img[non_maxima] = 0

    return img

def topN(arr: NdArray, n: int) -> Tuple[List, List]:
    """
    Extracts the indices of the top-N largest largest rhos and thetas in the accumulator array.
    """
    
    #? get the top N indices
    # NOTE: https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
    ind = np.argpartition(-arr, n, None)[:n]
    
    #? unravel to rho and theta indices
    # NOTE: https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
    return np.unravel_index(ind, arr.shape)
