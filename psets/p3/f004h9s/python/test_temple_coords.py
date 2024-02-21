#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Amittai Siavava (github: siavava)"

import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

#? 1. Load the two temple images and the points from data/some_corresp.npz

im1 = io.imread('../data/im1.png')
im2 = io.imread('../data/im2.png')
some_corresp = np.load('../data/some_corresp.npz')

pts1_corr = some_corresp['pts1']
pts2_corr = some_corresp['pts2']

#? 2. Run eight_point to compute F

# F = sub.eight_point(pts1_corr, pts2_corr, max(im1.shape))
F = sub.eight_point(pts1_corr, pts2_corr)  #! use M = 1

#? 3. Load points in image 1 from data/temple_coords.npz

temple_coords = np.load('../data/temple_coords.npz')
pts1 = temple_coords['pts1']


#? 4. Run epipolar_correspondences to get points in image 2

pts2 = sub.epipolar_correspondences(im1, im2, F, pts1)

#? 5. Compute the camera projection matrix P1

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E = sub.essential_matrix(F, K1, K2)

print(f"ESSENTIAL MATRIX: {E}")

print(f"{K2 = }")
P1 = np.hstack((K1, np.zeros((3,1))))

#? 6. Use camera2 to get 4 camera projection matrices P2

extrinsics = hlp.camera2(E)

P2s = [K2 @ extrinsics[:, :, i] for i in range(4)]
print(f"{P2s = }")

#? 7. Run triangulate using the projection matrices

pts3ds = [ sub.triangulate(P1, pts1, P2, pts2) for P2 in P2s ]

#? 8. Figure out the correct P2

invalid_count = [np.sum(p[:,2] < 0) for p in pts3ds]

best = np.argmin(invalid_count)
P2 = P2s[best]
pts3d = pts3ds[best]

print(f"{invalid_count = }")

# calculate reprojection error
pts1_reproj = P1 @ np.vstack((pts3d.T, np.ones((1, pts3d.shape[0]))))
pts1_reproj = pts1_reproj / pts1_reproj[2, :]
pts1_reproj = pts1_reproj[:2, :].T
reprojection_error1 = np.linalg.norm(pts1_reproj - pts1, axis=1).mean()
print(f"PTS1 ERROR: {reprojection_error1}")

pts2_reproj = P2 @ np.vstack((pts3d.T, np.ones((1, pts3d.shape[0]))))
pts2_reproj = pts2_reproj / pts2_reproj[2, :]
pts2_reproj = pts2_reproj[:2, :].T
reprojection_error2 = np.linalg.norm(pts2_reproj - pts2, axis=1).mean()
print(f"PTS2 ERROR: {reprojection_error2}")

#? 9. Scatter plot the correct 3D points

axis = plt.axes(projection='3d')
axis.set_zlim(1, 4)
axis.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])

# set axis labels
axis.set_xlabel('X')
axis.set_ylabel('Y')
axis.set_zlabel('Z')
plt.title('3D Points in Space')

plt.show()

#? 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz

R1 = np.eye(3)
print(f"{R1 = }")
t1 = np.zeros((3,1))
print(f"{t1 = }")
R2 = extrinsics[:,:,best][:,:3]
print(f"{R2 = }")
t2 = extrinsics[:,:,best][:, 3].reshape(3, 1)
print(f"{t2 = }")
np.savez('../data/extrinsics.npz', R1=R1, R2=R2, t1=t1, t2=t2)
