import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    """
        Lucas Kanade Affine Algorithm
        
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
    x1,y1,x2,y2 = rect

    # put your implementation here
    image, template = It1, It
    
    # TODO: splines for both images
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    
    template_spline = RectBivariateSpline(x, y, template.T)
    image_spline = RectBivariateSpline(x, y, image.T)

    # TODO: meshgrid
    x_mesh, y_mesh = np.meshgrid(np.arange(x1, x2+1, 1), np.arange(y1, y2+1, 1))     
    x_mesh, y_mesh = x_mesh.ravel(), y_mesh.ravel()                                         #? flatten
    
    #? evaluate the template spline at the meshgrid
    template_eval = template_spline.ev(x_mesh, y_mesh)
    
    # TODO: homogenize points
    points = np.vstack((x_mesh, y_mesh, np.ones(x_mesh.shape[0])))

    # TODO: jacobian matrix
    n = 2 * x_mesh.shape[0]
    jacobian = np.zeros((n, 6))

    jacobian[np.arange(0, n, 2), 0] = x_mesh
    jacobian[np.arange(1, n, 2), 1] = x_mesh
    
    jacobian[np.arange(0, n, 2), 2] = y_mesh
    jacobian[np.arange(1, n, 2), 3] = y_mesh

    jacobian[np.arange(0, n, 2), 4] = 1
    jacobian[np.arange(1, n, 2), 5] = 1
    
    jacobian = jacobian.reshape(n // 2, 2, 6)
    
    # TODO: iterate until maxIters OR convergence
    for _ in range(maxIters):

        # ? warp matrix
        M = np.array([
            [1.0+p[0], p[2],    p[4]],
            [p[1],    1.0+p[3], p[5]]
        ]).reshape((2, 3))
        
        
        #? find warped points
        warped_points = M @ points
        x_warped, y_warped = warped_points[0], warped_points[1]

        #? evaluate the image spline at the warped points
        image_eval = image_spline.ev(x_warped, y_warped)

        #? compute grad
        image_grad = np.stack([
            image_spline.ev(x_warped, y_warped, dx=1)[:, np.newaxis],
            image_spline.ev(x_warped, y_warped, dy=1)[:, np.newaxis]
        ], 2)

        #? error image
        error_image = (template_eval - image_eval).reshape((template_eval.shape[0], 1, 1))

        #? jacobian
        J = np.matmul(image_grad, jacobian)

        #? Hessian
        H = (np.transpose(J, (0, 2, 1)) @ J).sum(0)

        #? least squares solution
        b = (np.transpose(J, (0, 2, 1)) @ error_image).sum(0)

        #? desired change in parameters
        delta_p = np.linalg.pinv(H) @ b

        # TODO: update params
        p = p + delta_p
        
        #? converged yet?
        if np.linalg.norm(delta_p) < threshold:
            break

    # TODO: construct affine-warp matrix from the computed parameters
    M = np.array([
        [1.0+p[0],      p[2],       p[4]],
        [p[1],      1.0+p[3],       p[5]]
    ]).reshape((2, 3))
        
    return M
