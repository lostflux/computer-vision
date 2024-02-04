import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# TODO: Q3.4
def matchPics(I1, I2):
    """
    Function to match two images
    
    ## Parameters
    
    I1 : numpy.ndarray
        Image 1
    I2 : numpy.ndarray
        Image 2
    """

    #! hyperparameters -- vary to obtain the best results.    
    SIGMA = 0.15
    RATIO = 0.65

    #? Convert Images to GrayScale
    grayscale_1 = skimage.color.rgb2gray(I1)
    grayscale_2 = skimage.color.rgb2gray(I2)

    #? Detect Features in Both Images
    features_1 = corner_detection(grayscale_1, sigma=SIGMA)
    features_2 = corner_detection(grayscale_2, sigma=SIGMA)

    #? Obtain descriptors for the computed feature locations
    descriptors_1, locations_1 = computeBrief(grayscale_1, features_1)
    descriptors_2, locations_2 = computeBrief(grayscale_2, features_2)

    #? Match features using the descriptors
    matches = briefMatch(descriptors_1, descriptors_2, ratio=RATIO)

    return matches, locations_1, locations_2
