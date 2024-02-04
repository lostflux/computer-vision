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
    matches, locations_1, locations_2 = matchPics(cv_desk, cv_cover)
    
    #? pick points
    points_1 = np.take(locations_1, matches[:, 0], axis=0)
    points_2 = np.take(locations_2, matches[:, 1], axis=0)
    
    #? compute homography
    H2to1, inliers = computeH_ransac(points_1, points_2)

    #? composite images
    rescaled_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
    composite_img = compositeH(H2to1, cv_desk, rescaled_cover)

    cv2.imshow("image", composite_img)

    cv2.imwrite("../results/composite_img.jpg", composite_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    harry_potterrize()
