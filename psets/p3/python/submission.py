#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Amittai Siavava"

"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2
from scipy.signal import correlate

import helper




def eight_point(pts1, pts2, M=1):
    """
        Eight Point Algorithm
        
        Parameters
        ----------
        pts1 : points in image 1 (Nx2 matrix)
        pts2 : points in image 2 (Nx2 matrix)
        M : scalar value computed as max(H1,W1)
            
        Returns
        -------
        F : the fundamental matrix (3x3 matrix)
    """
    
    # TODO: 2.1
    
    #? divide each coordinate by M
    pts1 = pts1 / M
    pts2 = pts2 / M

    #? construct matrix A
    pts_count, _ = pts1.shape
    A = np.zeros((pts_count, 9))
    for p in range(pts_count):
        x1, y1 = pts1[p]
        x2, y2 = pts2[p]
        A[p] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

    #? compute the singular value decomposition of A
    _, _, V = np.linalg.svd(A)

    #? F is the last column of V
    F = V[-1].reshape(3, 3)

    #? enforce rank 2 constraint
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V
    
    #? refine F
    F = helper.refineF(F, pts1, pts2)

    #? unnormalize
    T = np.array(
        [
            [1/M, 0, 0],
            [0, 1/M, 0],
            [0,   0, 1]
        ])
    F = T.T @ F @ T
    
    return F




def epipolar_correspondences(im1, im2, F, pts1, window=20):
    """
        Epipolar Correspondences
        
        Parameters
        ----------
        im1 : image 1 (H1xW1 matrix)
        im2 : image 2 (H2xW2 matrix)
        F : fundamental matrix from image 1 to image 2 (3x3 matrix)
        pts1 : points in image 1 (Nx2 matrix)
        
        Returns
        -------
        pts2 : points in image 2 (Nx2 matrix)
    """
    
    #? convert images to grayscale
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    #? change pts1 datatype to int and convert points to homogeneous
    # pts1 = 
    pts1 = np.hstack( (pts1, np.ones((pts1.shape[0], 1))) )
    
    #? compute epipolar lines
    lines = (F @ pts1.T).T
    
    # print(f"{lines = }")
    
    #? pad
    pad = int(window // 2)
    # im1 = im1[:5, :5]
    # print(f"{im1 = }")
    im1 = np.pad(im1, pad, mode='constant', constant_values=0)
    # print(f"{im1 = }")
    im2 = np.pad(im2, pad, mode='constant', constant_values=0)
    
    # initialize points
    pts2 = np.zeros_like(pts1)
    
    # print(f"{pts1 = }")
    
    #? for each point
    for i in range(pts1.shape[0]):
        #? get window around point in first image
        
        #? get x, y coordinates (z should be 1)
        x, y, _ = pts1[i]
        x, y = int(x), int(y)
        
        #? define window_x and window_y, shifted by the initial pad
        # NOTE: sifted by initial pad
        #   example: with a pad of 2, original index 0 is now at index 2,
        #   and its window goes from index 0 to index 4 (x, x + 2*pad + 1)
        window1 = im1[y: y + 2*pad + 1, x: x + 2*pad + 1]
        
        #? compute disparity map
        disparities = np.ones(im2.shape[0] - 2*pad, dtype=np.float32) * np.inf
        
        for x in range(pad, im2.shape[0] - 2*pad):
            
            if x >= len(disparities):
                break
            
            #? compute corresponding y
            y = int( (-lines[i, 0] * x - lines[i, 2]) // lines[i, 1] )
            
            # find window
            window2 = im2[y : y + 2*pad + 1, x : x + 2*pad + 1]
            
            if window2.shape == window1.shape:
                disparities[x] = np.sum((window1 - window2)**2)
        
        #? find point in second image with minimum disparity
        # print(f"{disparities}")
        min_disparity = np.min(disparities)
        x2 = np.where(disparities == min_disparity)[-1]
        y2 = int( (-lines[i, 0] * x2[-1] - lines[i, 2]) // lines[i, 1] )
        # print(f"{pts2 = }")
        # print(f"{[x2[-1], y2, 1] = }")
        pts2[i] = [x2[-1], y2, 1]
    
    # print(f"{pts2 = }")
    return pts2[:, :2]





def essential_matrix(F, K1, K2):
    """
        Essential Matrix
        
        Parameters
        ----------
        F : the fundamental matrix (3x3 matrix)
        K1 : intrinsic camera matrix 1 (3x3 matrix)
        K2 : intrinsic camera matrix 2 (3x3 matrix)
        
        Returns
        -------
        E : the essential matrix (3x3 matrix)
    """
    
    # TODO: 2.3
    
    #? compute essential matrix
    E = K2.T @ F @ K1
    
    return E




"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    """
        Triangulation
        
        Parameters
        ----------
        
        P1: camera projection matrix 1 (3x4 matrix)
        pts1: points in image 1 (Nx2 matrix)
        P2: camera projection matrix 2 (3x4 matrix)
        pts2: points in image 2 (Nx2 matrix)
        
        Returns
        -------
        pts3d: 3D points in space (Nx3 matrix)
    """
    
    #? convert points to homogeneous coordinates
    if pts1.shape[1] == 2:
        pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        
    if pts2.shape[1] == 2:
        pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    
    #? initialize 3D points
    pts3d = np.zeros((pts1.shape[0], 3))
    
    #? for each point
    for i in range(pts1.shape[0]):
        
        #? construct (4x4) matrix A
        A = np.zeros((4, 4))
        A[0] = pts1[i, 0] * P1[2] - P1[0]
        A[1] = pts1[i, 1] * P1[2] - P1[1]
        A[2] = pts2[i, 0] * P2[2] - P2[0]
        A[3] = pts2[i, 1] * P2[2] - P2[1]
        
        #? compute the singular value decomposition of A
        _, _, V = np.linalg.svd(A)
        
        #? the 3D point is the last column of V. Normalize by z
        pts3d[i] = V[-1, :3] / V[-1, 3]
    
    return pts3d




def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
        Image Rectification
        
        Parameters
        ----------
        K1, K2: camera matrices (3x3 matrix)
        R1, R2: rotation matrices (3x3 matrix)
        t1, t2: translation vectors (3x1 matrix)
        
        Returns
        -------
        M1, M2: rectification matrices (3x3 matrix)
        K1p, K2p: rectified camera matrices (3x3 matrix)
        R1p, R2p: rectified rotation matrices (3x3 matrix)
        t1p, t2p: rectified translation vectors (3x1 matrix)
    """
    
    #? compute optical centers
    c1 = - np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = - np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    
    #? compute r1, r2, r3
    diff = c1 - c2
    r1 = diff / np.linalg.norm(diff)
    r2 = np.cross(R1[-1, :].reshape(3, 1), r1, axis=0)
    r3 = np.cross(r2, r1, axis=0)
    
    R1p = np.hstack((r1, r2, r3)).T
    R2p = R1p
    
    print(f"{R1p = }")
    
    #? compute new camera matrices
    K1p = K2
    K2p = K2
    
    print(f"{K1p = }")
    
    #? compute new translation vectors
    t1p = -R1p @ c1
    t2p = -R2p @ c2
    
    #? compute rectification matrices
    M1 = K1p @ R1p @ np.linalg.inv(K1 @ R1)
    M2 = K2p @ R2p @ np.linalg.inv(K2 @ R2)
    
    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p




def get_disparity(im1, im2, max_disp, win_size):
    """
        Disparity Map
        
        Inputs
        ------
        im1 : image 1 (H1xW1 matrix)
        im2 : image 2 (H2xW2 matrix)
        max_disp : scalar maximum disparity value
        win_size : scalar window size value
        
        Returns
        -------
        dispM : disparity map (H1xW1 matrix)
    """
    
    #? initialize disparity map
    dispM = np.zeros_like(im1, dtype=float)
    
    #? pad images
    pad = win_size // 2
    im1 = np.pad(im1, pad, mode='constant', constant_values=0)
    im2 = np.pad(im2, pad, mode='constant', constant_values=0)
    
    #? for each pixel in the first image
    for i in range(pad, im1.shape[0]-pad):
        for j in range(pad, im1.shape[1]-pad):
            
            #? get window around pixel in first image
            window1 = im1[i-pad:i+pad+1, j-pad:j+pad+1]
            
            #? compute disparity map
            disparity_map = np.ones(max_disp+1) * np.inf
            for d in range(max_disp+1):
                if j-d-pad >= 0:
                    window2 = im2[i-pad:i+pad+1, j-d-pad:j-d+pad+1]
                    disparity_map[d] = np.sum((window1 - window2)**2)
            
            #? find disparity with minimum disparity
            min_disparity = np.min(disparity_map)
            dispM[i-pad, j-pad] = np.where(disparity_map == min_disparity)[0][0]
    
    return dispM
    
    


def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
        Depth Map
        
        Inputs
        ------
        dispM : disparity map (H1xW1 matrix)
        K1, K2 : camera matrices (3x3 matrix)
        R1, R2 : rotation matrices (3x3 matrix)
        t1, t2 : translation vectors (3x1 matrix)
        
        Returns
        -------
        depthM : depth map (H1xW1 matrix)
    """
    
    #? optical centers
    c1 = - np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = - np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    
    #? compute baseline
    b = np.linalg.norm(c1 - c2)
    
    #? compute focal length
    f = K1[0, 0]
    
    #? compute depth map
    depthM = np.zeros_like(dispM, dtype=float)
    depthM[dispM != 0] = b * f / dispM[dispM != 0]
    
    return depthM




def estimate_pose(x, X):
    """
        Camera Matrix Estimation
        
        Inputs
        ------
        x : 2D points (Nx2 matrix)
        X : 3D points (Nx3 matrix)
        
        Returns
        -------
        P : camera matrix (3x4 matrix)
    """
    
    #? convert points to homogeneous coordinates
    if x.shape[1] == 2:
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        
    if X.shape[1] == 3:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    
    #? compute matrix A
    A = np.zeros((2*x.shape[0], 12))
    for i in range(x.shape[0]):
        A[2*i] = np.hstack((X[i], np.zeros(4), -x[i, 0]*X[i]))
        A[2*i+1] = np.hstack((np.zeros(4), X[i], -x[i, 1]*X[i]))
    
    #? compute the singular value decomposition of A
    _, _, V = np.linalg.svd(A)
    
    #? the camera matrix is the last column of V
    P = V[-1].reshape(3, 4)
    
    return P




def estimate_params(P):
    """
        Camera Parameter Estimation
        
        Inputs
        ------
        P : camera matrix (3x4 matrix)
        
        Returns
        -------
        K : camera intrinsics (3x3 matrix)
        R : camera extrinsics rotation (3x3 matrix)
        t : camera extrinsics translation (3x1 matrix)
    """
    
    #? compute camera center using SVD
    _, _, V = np.linalg.svd(P)
    c = V[-1]
    c = c / c[-1]     #! norm by z
    c = c[:3]         #! convert to heterogenous
    
    #? compute camera intrinsics K and rotation R using QR decomposition
    K, R = np.linalg.qr(P[:, :3])
    
    #? compute camera extrinsics translation t
    t = -R @ c
    
    return K, R, t
    
    
    


##############################################
# TESTS
##############################################

def test_eight_point():
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')
    
    points = np.load('../data/some_corresp.npz')
    # print(f"{points = }")
    pts1 = points['pts1']
    pts2 = points['pts2']
    
    M = max(im1.shape)
    F = eight_point(pts1, pts2, M)
    
    # convert images to homogeneous
    helper.displayEpipolarF(im1, im2, F)
    
def test_epipolar_correspondences():
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')
    
    points = np.load('../data/some_corresp.npz')
    pts1 = points['pts1']
    pts2 = points['pts2']
    
    M = max(im1.shape)
    F = eight_point(pts1, pts2, M)
    
    # predicted_pts2 = epipolar_correspondences(im1, im2, F, pts1)
    
    helper.epipolarMatchGUI(im1, im2, F)
    # print(pts2)

if __name__ == "__main__":
    
    #? test eight_point
    # test_eight_point()
    
    #? test epipolar_correspondences
    test_epipolar_correspondences()
