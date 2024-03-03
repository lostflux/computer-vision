import numpy as np
import cv2


def computeH(x1, x2):
    """
    Compute the homography between two sets of points

    Parameters
    ----------
    dest : array
        Array of N points in the destination image.
    source : array
        Array of N points in the source image.
        
    ! NOTE: 2 to 1 not 1 to 2 (had a bug here, fixed it)

    Returns
    -------
    H2to1 : array
        Homography matrix.
    """

    # # TODO: Q3.6 (computeH)
    
    dest = x1
    source = x2

    N, _ = dest.shape
    A = np.zeros((2*N, 9))

    for i in range(N):
        x_dest, y_dest = dest[i, 0], dest[i, 1]
        x_source, y_source = source[i, 0], source[i, 1]
        
        A[2*i, :] = [-x_source, -y_source, -1, 0, 0, 0, x_dest*x_source, y_dest*x_source, x_dest]
        A[2*i + 1, :] = [0, 0, 0, -x_source, -y_source, -1, x_dest*y_source, y_dest*y_source, y_dest]
        
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

	# TODO: Q3.7 (computeH_norm)

    #? Compute the centroid of the points
    centroid1 = np.mean(x1, axis=0)
    centroid2 = np.mean(x2, axis=0)

    #? Shift the origin of the points to the centroid
    x1_shifted = x1 - centroid1
    x2_shifted = x2 - centroid2

    #? Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_dist1 = np.max(np.linalg.norm(x1_shifted, axis=1))
    max_dist2 = np.max(np.linalg.norm(x2_shifted, axis=1))
    
    scale1 = np.sqrt(2) / max_dist1
    scale2 = np.sqrt(2) / max_dist2
    
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
    H2to1 = np.linalg.inv(T1) @ H_normalized @ T2

    return H2to1

def computeH_ransac(x1, x2, threshold=5, iterations=100):
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
    
    # TODO: Q3.8 (computeH_ransac)

    N, _ = x1.shape
    max_inliers = 0
    
    best_inliers = None
    
    #? flip image coordinates
    #? (x, y) -> (y, x)
    x1 = np.fliplr(x1)
    x2 = np.fliplr(x2)

    x2_homogeneous = np.vstack((x2.T, np.ones(N)))      #! transpose and append row of 1s
    
    for iteration in range(1, iterations+1):
        
        #? Randomly select 4 points
        p = np.random.choice(N, 4, replace=False)
        x1_samples = x1[p, :]
        x2_samples = x2[p, :]

        #? Compute the homography
        H = computeH_norm(x1_samples, x2_samples)

        #? Compute the transformed points
        x2_transformed = H @ x2_homogeneous             #! apply homography
        x2_transformed /= x2_transformed[2]             #! normalize by z

        #? Compute the error
        distances = np.linalg.norm(x2_transformed[:2] - x1.T, axis=0)

        #? Find the inliers
        inliers = distances < threshold

        if iteration % 10 == 0:
            print(f"{iteration:5d}: {max_inliers:5d}")

        #? Determine the number of inliers
        num_inliers = np.sum(inliers)

        #? Update the best inliers if necessary
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers

    print(f" BEST: {max_inliers:5d}")
    
    #? Compute better homography estimate
    #? using ALL the inliers.
    indices = np.where(best_inliers)
    x1_inliers = x1[indices]
    x2_inliers = x2[indices]
    bestH2to1 = computeH_norm(x1_inliers, x2_inliers)
    
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
    
    # TODO: Q3.9 (compositeH)

    #? Invert the homography matrix
    # TODO: Should I be inverting this??????????
    H = H2to1
    # H = np.linalg.inv(H2to1)

    #? Create a mask of the same size as the template
    mask = np.ones(template.shape[:2])

    #? Warp the mask and the template by the appropriate homography
    height, width, _ = img.shape
    warped_mask = cv2.warpPerspective(mask, H, (width, height))
    warped_template = cv2.warpPerspective(template, H, (width, height))

    #? Use the mask to combine the warped template and the image
    composite_img = np.zeros_like(img)
    #! set each channel of the composite image to the warped image
    #!    if the mask is 1,
    #!    else set it to the template
    for c in range(3):
        composite_img[:, :, c] = warped_template[:, :, c] * warped_mask + img[:, :, c] * (1 - warped_mask)

    return composite_img
