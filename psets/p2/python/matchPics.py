#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import plotMatches
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, SIGMA = 0.15, RATIO = 0.65):
    """
    Function to match two images
    
    ## Parameters
    
    I1 : numpy.ndarray
        Image 1
    I2 : numpy.ndarray
        Image 2
    """

    # TODO: Q3.4
    
    #! hyperparameters -- vary to obtain the best results.    

    #? Convert Images to GrayScale if needed
    if len(I1.shape) == 3:
        gray1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    #? Detect Features in Both Images
    features1 = corner_detection(gray1, sigma=SIGMA)
    features2 = corner_detection(gray2, sigma=SIGMA)

    #? Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(gray1, features1)
    desc2, locs2 = computeBrief(gray2, features2)

    #? Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio=RATIO)

    return matches, locs1, locs2

if __name__ == "__main__":
    #? Read the images
    image1_path = "../data/cv_cover.jpg"
    image2_path = "../data/cv_desk.png"
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    #? Match features between the two images
    matches, locs1, locs2 = matchPics(image1, image2, RATIO=0.8)

    #? Plot matches
    plotMatches(image1, image2, matches, locs1, locs2)
