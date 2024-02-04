import numpy as np
import cv2


def computeH(x1, x2):
    """
    Compute the homography between two sets of points

    Parameters
    ----------
    x1 : array
        Array of N points in the destination image.
    x2 : array
        Array of N points in the source image.
        
    ! NOTE: 2 to 1 not 1 to 2 (had a bug here, fixed it)

    Returns
    -------
    H2to1 : array
        Homography matrix.
    """

    # TODO: Q3.6

    N = x1.shape[1]
    A = np.zeros((2*N, 9))
    
    # print(f"{x1.shape = }")
    # print(f"{x2.shape = }")

    for i in range(N):
        X1, Y1 = x1[i, 0], x1[i, 1]
        X2, Y2 = x2[i, 0], x2[i, 1]
        A[2*i, :] = [-X1, -Y1, -1, 0, 0, 0, X1*X2, Y1*X2, X2]
        A[2*i + 1, :] = [0, 0, 0, -X1, -Y1, -1, X1*Y2, Y1*Y2, Y2]

    _, _, V = np.linalg.svd(A)
    H2to1 = V[-1].reshape(3, 3)

    return H2to1


def computeH_norm(x1, x2):
    """
    Compute the normalized homography between two sets of points

    Parameters
    ----------
    x1 : array
        Array of N points in the destination image.
    x2 : array
        Array of N points in the source image.
        
    ! NOTE: 2 to 1 not 1 to 2 (had a bug here, fixed it)

    Returns
    -------
    H2to1 : array
        Homography matrix.
    """

	# TODO: Q3.7

    #? Compute the centroid of the points
    centroid1 = np.mean(x1, axis=0)
    centroid2 = np.mean(x2, axis=0)
    
    #! Debugging
    #? flattten centroids to 1D
    # centroid1 = centroid1.flatten()
    # centroid2 = centroid2.flatten()
    
    # print(f"{centroid1 = }")
    # print(f"{centroid2 = }")

    #? Shift the origin of the points to the centroid
    x1_shifted = x1 - centroid1
    x2_shifted = x2 - centroid2

    #? Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    scale1 = np.sqrt(2) / np.max(np.linalg.norm(x1_shifted, axis=0))
    scale2 = np.sqrt(2) / np.max(np.linalg.norm(x2_shifted, axis=0))
    x1_normalized = x1_shifted * scale1
    x2_normalized = x2_shifted * scale2

    #? Similarity transform 1
    T1 = np.array([
        [scale1, 0, -scale1 * centroid1[0]],
        [0, scale1, -scale1 * centroid1[1]],
        [0, 0, 1]
    ])

    #? Similarity transform 2
    T2 = np.array([
        [scale2, 0, -scale2 * centroid2[0]],
        [0, scale2, -scale2 * centroid2[1]],
        [0, 0, 1]
    ])

    #? Compute homography
    H_normalized = computeH(x1_normalized, x2_normalized)

    #? Denormalization
    # NOTE: @ is np shorthand for matrix multiplication
    H2to1 = np.linalg.inv(T2) @ H_normalized @ T1 

    return H2to1

def computeH_ransac(x1, x2, threshold=10, iterations=10000):
    """
    Compute the best fitting homography given a list of matching points

    Parameters
    ----------
    x1 : array
        Array of N points in the source image.
    x2 : array
        Array of N points in the destination image.
    threshold : float
        The threshold used to determine inliers.

    Returns
    -------
    bestH2to1 : array
        Best fitting homography matrix.
    inliers : array
        Boolean array indicating which points are inliers.
    """
    
    # TODO: Q3.8
    
    # x1 = 

    N, _ = x1.shape
    max_inliers = 0
    
    #? init bestH2to1 to ID
    bestH2to1 = np.eye(3)
    best_inliers = np.zeros(N)
    
    
    x1 = np.flip(x1, axis=1)
    x2 = np.flip(x2, axis=1)

    for iteration in range(iterations):
        
        #? Randomly select 4 points
        indices = np.random.choice(N, 4, replace=False)
        x1_samples = x1[indices, :]
        x2_samples = x2[indices, :]
        
        # print(f"{x1_samples = }")
        # print(f"{x1_samples.shape = }")

        #? Compute the homography
        H = computeH_norm(x1_samples, x2_samples)

        #? Compute the transformed points
        x2_homogeneous = np.vstack((x2.T, np.ones(N)))  #! append row of 1s
        # print(f"{x2_homogeneous = }")
        x2_transformed = np.dot(H, x2_homogeneous)    #! apply homography
        x2_transformed /= x2_transformed[2]           #! normalize by z

        #? Compute the error
        distances = np.linalg.norm(x2_transformed[:2] - x1.T, axis=0)
        
        # print(f"{distances = }")
        # print(f"{distances.shape = }")
        
        #? Find the inliers
        inliers = distances < threshold
        
        print(f"{iteration:5d}: {np.sum(inliers):5d}")

        #? Determine the number of inliers
        num_inliers = np.sum(inliers)

        #? Update the best inliers if necessary
        if 0 < num_inliers and num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            bestH2to1 = H

    #? Recompute the homography using all best inliers
    print(f"best inliers: {np.sum(best_inliers)}")
    # points = np.nonzero(best_inliers)                         # pick 1's from best_inliers
    # print(f"{best_inliers = }")
    # print(f"{points = }")
    # bestH2to1 = computeH_norm(x1[points, :], x2[points, :])   # compute H using best inliers


    return bestH2to1, best_inliers

def compositeH(H2to1, template, img):
    """
    Create a composite image after warping the template image on top
    of the image using the homography
    
    NOTE:   H2to1 is from the image to the template;
	        For warping the template to the image, we need to invert it.

    Parameters
    ----------
    H2to1 : array
        Homography matrix.
    template : array
        The template image.
    img : array
        The image.

    Returns
    -------
    composite_img : array
        The composite image.
    """
    
    # TODO: Q3.9

    #? Invert the homography matrix
    # TODO: Should I be inverting this??????????
    # H = H2to1
    H = np.linalg.inv(H2to1)

    #? Create a mask of the same size as the template
    mask = np.ones(img.shape[:2], img.dtype)

    #? Warp the mask and the image by the appropriate homography
    height, width, _ = template.shape
    warped_mask = cv2.warpPerspective(mask, H, (width, height))
    warped_img = cv2.warpPerspective(img, H, (width, height))

    #? Use the mask to combine the warped template and the image
    composite_img = np.zeros_like(template)
    #! set each channel of the composite image to the warped image
    #!    if the mask is 1,
    #!    else set it to the template
    for c in range(3):
        composite_img[:, :, c] = warped_img[:, :, c] * warped_mask + template[:, :, c] * (1 - warped_mask)

    return composite_img

