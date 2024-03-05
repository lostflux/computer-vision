#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Matthews-Baker (Inverse Compositional) Alignment
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.typing import NDArray

def InverseCompositionAffine(It, It1, rect) -> NDArray[np.float64]:
    """
        Inverse Compositional Algorithm
        
        Parameters
        ----------
        - It: template image
        - It1: Current image
        - rect: Current position of the object
            (top left, bot right coordinates: x1, y1, x2, y2)
        
        Returns
        -------
        - M: the Affine warp matrix [2x3 numpy array]
    """

    # TODO: set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))
    x1, y1, x2, y2 = rect

    # put your implementation here
    image, template = It1, It

    # TODO: splines for both images
    x_range = np.arange(0, template.shape[1])
    y_range = np.arange(0, template.shape[0])

    image_spline = RectBivariateSpline(x_range, y_range, image.T)
    template_spline = RectBivariateSpline(x_range, y_range, It.T)

    # TODO: meshgrid for the template image
    x_mesh, y_mesh = np.meshgrid(np.arange(x1, x2+1, 1), np.arange(y1, y2+1, 1))     
    x_mesh, y_mesh = x_mesh.ravel(), y_mesh.ravel()                                         #? flatten

    #? evaluate the template spline at the meshgrid
    template_eval = template_spline.ev(x_mesh, y_mesh)
    
    # TODO: template gradient
    template_grad = np.stack([
        template_spline.ev(x_mesh, y_mesh, dx=1)[:, np.newaxis],
        template_spline.ev(x_mesh, y_mesh, dy=1)[:, np.newaxis]
    ], 2)       

    # TODO: jacobian matrix
    n = 2 * x_mesh.shape[0]

    jacobian = np.array([
        [[x, 0, y, 0, 1, 0],
         [0, x, 0, y, 0, 1]]
        for x, y in zip(x_mesh, y_mesh)
    ]).reshape(n // 2, 2, 6)
    

    # TODO: hessian matrix and its pseudo-inverse
    J = template_grad @ jacobian
    H = (np.transpose(J, (0, 2, 1)) @ J).sum(0)
    H_inv = np.linalg.pinv(H)                                       #? pseudo-inverse

    # TODO: homogenize points
    points = np.vstack((x_mesh, y_mesh, np.ones(x_mesh.shape[0])))
    
    W = np.eye(3)                                                   #? start with identity warp
    
    # TODO: iterate until maxIters OR convergence
    for _ in range(maxIters):

        #? compute warped points
        warped_points = W @ points

        #? interpolate to compute intensity of warped points
        x_mesh, y_mesh = warped_points[0], warped_points[1]
        image_eval = image_spline.ev(warped_points[0], warped_points[1])
        
        #? compute the error image
        error_image = (image_eval - template_eval).reshape(template_eval.shape[0], 1, 1)   

        #? least squares solution
        b = (np.transpose(J, (0, 2, 1)) @ error_image).sum(0)

        #? desired change in parameters
        delta_p = H_inv @ b
        
        #? desired change in warp matrix (with warp)
        W_delta = np.array([
            [1.0 + delta_p[0].item(),          delta_p[2].item(),     delta_p[4].item()],
            [      delta_p[1].item(),    1.0 + delta_p[3].item(),     delta_p[5].item()],
            [                      0,                          0,                     1]
        ], dtype=np.float64)


        # TODO: update params
        W = W @ np.linalg.inv(W_delta)
        
        #? converged yet?
        if np.linalg.norm(delta_p) < threshold:
            break
    
    #? keep only the first 2 rows
    M = W[:2]
    
    return M
