import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2):
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
    SIGMA = 0.15
    RATIO = 0.65

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
