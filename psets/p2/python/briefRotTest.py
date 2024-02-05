#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rotation Tests for BRIEF Descriptor
"""

import numpy as np
import cv2
from matchPics import matchPics
import scipy
import skimage
from helper import plotMatches
import matplotlib.pyplot as plt


# TODO: Q3.5 (BRIEF Rotation Test)

def rotation_test():
    """
    Function to test the rotation invariance of the BRIEF descriptor
    """
    
    #? read image, convert to grayscale if needed
    image_path = "../data/cv_cover.jpg"
    # image = skimage.color.rgb2gray(cv2.imread(image_path))
    image = cv2.imread(image_path)

    X, y = [], []

    for deg in range(10, 360, 10):

        #? Rotate Image
        image_rotated = scipy.ndimage.rotate(image, deg, reshape=False)
        
        #? Compute features, descriptors and Match features
        matches, locations_1, locations_2 = matchPics(image, image_rotated)
        match_count, _ = matches.shape

        #? Update histogram
        X.append(deg)
        y.append(match_count)
        
        if deg in (90, 180, 270):
            print(f"matches at {deg:3d} degrees: {match_count}")
            plotMatches(image, image_rotated, matches, locations_1, locations_2)
            
    #? display histogram
    plt.bar(X, y)
    plt.xlabel("Rotation (degrees)")
    plt.ylabel("Number of Matches")
    plt.title("Number of Matches vs Rotation")
    plt.show()
    
if __name__ == "__main__":
    rotation_test()
