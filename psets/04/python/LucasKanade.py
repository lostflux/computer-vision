#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Lucas-Kanade algorithm
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Tuple

def LucasKanade(It, It1, rect) -> Tuple[float, float]:
    """
        Lucas Kanade Algorithm
        
        Parameters
        ----------
        It: template image
        It1: Current image
        rect: Current position of the object
        (top left, bot right coordinates: x1, y1, x2, y2)
        
        Returns
        -------
        p: movement vector dx, dy
    """
    
    # TODO: set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)          
    x1, y1, x2, y2 = rect

    # put your implementation here
    image, template = It1, It
    
    # TODO: jacobian matrix
    jacobian = np.eye(2)                                            #! jacobian is just IDENTITY
    
    # TODO: splines for both images
    x_range = np.arange(0, It.shape[1])
    y_range = np.arange(0, It.shape[0])
    
    template_spline = RectBivariateSpline(x_range, y_range, template.T)
    image_spline = RectBivariateSpline(x_range, y_range, image.T)

    # TODO: meshgrid for the template image
    xt, yt = np.meshgrid(np.arange(x1, x2+1, 1), np.arange(y1, y2+1, 1))     
    xt, yt = xt.ravel(), yt.ravel()

    # TODO: evaluate template spline at the meshgrid
    template_eval = template_spline.ev(xt, yt)

    # TODO: iterate until maxIters OR convergence
    for _ in range(maxIters):
        
        #? find the warped points
        xi, yi = xt + p[0], yt + p[1]
        
        #? evaluate the image spline at the warped points
        image_eval = image_spline.ev(xi, yi)

        #? error image
        b = template_eval - image_eval
        
        #? image gradients
        image_grad = np.stack([
            image_spline.ev(xi, yi, dx=1),
            image_spline.ev(xi, yi, dy=1)
        ], 1)

        # TODO: calculate the least squares change for delta_p
        H = image_grad # @ jacobian                                 #! jacobian is just IDENTITY
        delta_p = np.linalg.lstsq(H, b, rcond=None)[0]
        
        # TODO: update the parameters
        p += delta_p
        
        # TODO: check if within threshold
        if np.linalg.norm(delta_p) < threshold:
            break

    return p[0], p[1]
