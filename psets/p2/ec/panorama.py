#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from matchPics import matchPics
from planarH import compositeH, computeH_ransac

def panoramize(left, right, key=0):
    """
    Function to compute the panorama of two images

    Parameters
    ----------
    left : numpy.ndarray
        Left image
    right : numpy.ndarray
        Right image

    Returns
    -------
    panorama : numpy.ndarray
        Panorama of the two images
    """
    
    # TODO: Q4.2x
    
    #? find matches
    matches, locs1, locs2 = matchPics(left, right)
    
    #? pick points
    points1 = np.take(locs1, matches[:, 0], axis=0)
    points2 = np.take(locs2, matches[:, 1], axis=0)
    
    #? compute homography
    H2to1, inliers = computeH_ransac(points1, points2)
    print(f"{H2to1 = }")
    
    #? test warp
    warped_right = cv2.warpPerspective(right, H2to1, (left.shape[1] + right.shape[1], left.shape[0]))
    cv2.imwrite(f"./results/warped-right-{key}.jpg", warped_right)
    cv2.imshow("Warped Right", warped_right)
    
    #? composite images
    # panorama = compositeH(H2to1, right, left)
    panorama = warped_right.copy()
    panorama[0:left.shape[0], 0:left.shape[1]] = left
    cv2.imshow("Panorama", panorama)
    cv2.imwrite(f"./results/panorama-{key}.jpg", panorama)
    cv2.waitKey(0)     #! don't close window until a key is pressed
    return panorama

if __name__ == "__main__":
    
    """ IMAGE 0 """
    # # ? Read the images
    left = cv2.imread("./data/pano_left.jpg")
    right = cv2.imread("./data/pano_right.jpg")
    
    #? Compute the panorama
    panorama = panoramize(left, right, 0)
    
    
    ############################################################
    
    
    """ IMAGE 1 """
    
    # # ? Read the images
    # left = cv2.imread("./data/left.jpg")
    # right = cv2.imread("./data/middle.jpg")
    
    # #? Compute the panorama
    # panorama = panoramize(left, right, 1)
    
    
    ############################################################
    
    
    """ IMAGE 2 """
    
    # # ? Read the images
    # left = cv2.imread("./data/middle.jpg")
    # right = cv2.imread("./data/right.jpg")
    
    # #? Compute the panorama    
    # panorama = panoramize(left, right, 2)
    
    
    ############################################################
    
        
    """ IMAGE 3 """
    
    # # ? Read the images
    # left = cv2.imread("./data/left.jpg")
    # right = cv2.imread("./results/panorama-2.jpg")
    
    # #? Compute the panorama
    # panorama = panoramize(left, right, 3)
    
    
    ############################################################
    
    
    #? Display the panorama
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)     #! don't close window until a key is pressed
    cv2.destroyAllWindows()
