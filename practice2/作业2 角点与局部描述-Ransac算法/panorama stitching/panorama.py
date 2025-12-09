"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/27/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above. If you use convolve(), remember to
        specify zero-padding to match our equations, for example:

            out_image = convolve(in_image, kernel, mode='constant', cval=0)

        You can also use for nested loops compute M and the subsequent Harris
        corner response for each output pixel, intead of using convolve().
        Your implementation of conv_fast or conv_nested in HW1 may be a
        useful reference!

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # 1. Compute x and y derivatives (I_x, I_y) of an image
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    # 2. Compute products of derivatives (I_x^2, I_y^2, I_xy) at each pixel
    Ixx = dx * dx
    Iyy = dy * dy
    Ixy = dx * dy
    
    # 3. Compute matrix M at each pixel using window function
    # Sum over window: w(x,y) * [I_x^2, I_xy; I_xy, I_y^2]
    Sxx = convolve(Ixx, window, mode='constant', cval=0)
    Syy = convolve(Iyy, window, mode='constant', cval=0)
    Sxy = convolve(Ixy, window, mode='constant', cval=0)
    
    # 4. Compute corner response R = Det(M) - k(Trace(M)^2)
    # Det(M) = Sxx * Syy - Sxy^2
    # Trace(M) = Sxx + Syy
    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    response = det_M - k * (trace_M ** 2)
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        In this case of normalization, if a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []

    ### YOUR CODE HERE
    # Normalize the patch to have mean 0 and std 1
    patch_flat = patch.flatten()
    mean = np.mean(patch_flat)
    std = np.std(patch_flat)
    
    # If denominator is zero, divide by 1 instead
    if std == 0:
        std = 1
    
    # Normalize: (x - mean) / std
    normalized = (patch_flat - mean) / std
    
    feature = normalized
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


import heapq
def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be strictly smaller
    than the threshold (not equal to). Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

        The Scipy function cdist calculates Euclidean distance between all
        pairs of inputs
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    M = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    # For each descriptor in desc1, find the closest and second-closest in desc2
    for i in range(M):
        # Get distances from desc1[i] to all descriptors in desc2
        distances = dists[i, :]
        
        # Sort to find closest and second-closest
        sorted_indices = np.argsort(distances)
        closest_idx = sorted_indices[0]
        second_closest_idx = sorted_indices[1]
        
        closest_dist = distances[closest_idx]
        second_closest_dist = distances[second_closest_idx]
        
        # Match if ratio is strictly smaller than threshold
        if second_closest_dist > 0 and closest_dist / second_closest_dist < threshold:
            matches.append([i, closest_idx])
    
    matches = np.array(matches)
    ### END YOUR CODE

    return matches


def fit_affine_matrix(p1, p2):
    """ 
    Fit affine matrix such that p2 * H = p1. First, pad the descriptor vectors
    with a 1 using pad() to convert to homogeneous coordinates, then return
    the least squares fit affine matrix in homogeneous coordinates.

    Hint:
        You can use np.linalg.lstsq function to solve the problem. 

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None 
        to suppress deprecation warnings, and match the autograder.

    Args:
        p1: an array of shape (M, P) holding descriptors of size P about M keypoints
        p2: an array of shape (M, P) holding descriptors of size P about M keypoints

    Return:
        H: a matrix of shape (P+1, P+1) that transforms p2 to p1 in homogeneous
        coordinates
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE
    H, residuals, rank, s = np.linalg.lstsq(p2, p1, rcond=None)
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation:

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers via Euclidean distance
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Update max_inliers as a boolean array where True represents the keypoint
    at this index is an inlier, while False represents that it is not an inlier.

    Hint:
        You can use np.linalg.lstsq function to solve the problem. 

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None 
        to suppress deprecation warnings, and match the autograder.

        You can compute elementwise boolean operations between two numpy arrays,
        and use boolean arrays to select array elements by index:
        https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing 

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N, dtype=bool)
    n_inliers = 0

    # RANSAC iteration start
    
    # Note: while there're many ways to do random sampling, please use
    # `np.random.shuffle()` followed by slicing out the first `n_samples`
    # matches here in order to align with the auto-grader.
    # Sample with this code:
    '''
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        sample1 = pad(keypoints1[samples[:,0]])
        sample2 = pad(keypoints2[samples[:,1]])
    '''

    ### YOUR CODE HERE
    
    for i in range(n_iters):
        # Use np.random.shuffle as specified in the comments
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        sample1 = pad(keypoints1[samples[:,0]])
        sample2 = pad(keypoints2[samples[:,1]])

        H, residuals, rank, s = np.linalg.lstsq(sample2, sample1, rcond=None)
        H[:, 2] = np.array([0, 0, 1])

        output = np.dot(matched2, H)
        inliersArr = np.linalg.norm(output-matched1, axis=1)**2 < threshold
        inliersCount = np.sum(inliersArr)

        if inliersCount > n_inliers:
            max_inliers = inliersArr.copy()
            n_inliers = inliersCount

    # 迭代完成，拿最大数目的匹配点对进行估计变换矩阵
    H, residuals, rank, s = np.linalg.lstsq(matched2[max_inliers], matched1[max_inliers], rcond=None)
    H[:, 2] = np.array([0, 0, 1])

    ### END YOUR CODE
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten block of histograms into a 1D feature vector
        Here, we treat the entire patch of histograms as our block
    4. Normalize flattened block by L2 norm
        Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    for i in range(rows):
        for j in range(cols):
            # Get gradient magnitude and angle for this cell
            G_cell = G_cells[i, j]
            theta_cell = theta_cells[i, j]
            
            # Compute histogram for this cell
            for bin_idx in range(n_bins):
                # Bin range: [bin_idx * degrees_per_bin, (bin_idx + 1) * degrees_per_bin)
                bin_start = bin_idx * degrees_per_bin
                bin_end = (bin_idx + 1) * degrees_per_bin
                
                # Find pixels in this bin (handle wrap-around at 180 degrees)
                if bin_idx == n_bins - 1:
                    # Last bin includes 180 degrees
                    mask = (theta_cell >= bin_start) & (theta_cell <= bin_end)
                else:
                    mask = (theta_cell >= bin_start) & (theta_cell < bin_end)
                
                # Sum gradient magnitudes in this bin
                cells[i, j, bin_idx] = np.sum(G_cell[mask])
    
    # Flatten block of histograms into 1D feature vector
    block = cells.flatten()
    
    # Normalize by L2 norm
    l2_norm = np.linalg.norm(block)
    if l2_norm > 0:
        block = block / l2_norm
    ### YOUR CODE HERE

    return block

