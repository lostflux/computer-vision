#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Harry Potterize two images
"""

import numpy as np
import cv2
import skimage.io 
import skimage.color

from matchPics import matchPics
from planarH import compositeH, computeH_ransac

def harry_potterrize():
    """
        harry_poterrize cv_cover.jpg and cv_desk.png
    """
    # TODO: Q3.9
    
    cv_cover = cv2.imread("../data/cv_cover.jpg")
    cv_desk = cv2.imread("../data/cv_desk.png")
    hp_cover = cv2.imread("../data/hp_cover.jpg")

    #? find matches
    matches, locs1, locs2 = matchPics(cv_desk, cv_cover)
    
    #? pick points
    points1 = np.take(locs1, matches[:, 0], axis=0)
    points2 = np.take(locs2, matches[:, 1], axis=0)
    
    #? compute homography
    H2to1, inliers = computeH_ransac(points1, points2)
    print(f"{H2to1 = }")
    
    #? test warp
    warped_cover = cv2.warpPerspective(hp_cover, H2to1, (cv_desk.shape[1], cv_desk.shape[0]))
    cv2.imshow("Warped Cover", warped_cover)
    
    #? rescale the cover
    rescaled_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
    

    #? composite images
    composite_img = compositeH(H2to1, rescaled_cover, cv_desk)

    cv2.imshow("Composite Image", composite_img)

    cv2.imwrite("../results/composite-image.jpg", composite_img)
    cv2.waitKey(0)     #! don't close window until a key is pressed

if __name__ == "__main__":
    harry_potterrize()
