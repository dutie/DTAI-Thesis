"""
thesis.oit
~~~~~~~~~~~

Orientation Invariant Techniques

:author: Mats Fockaert
:copyright: ...
:license: ...

"""

import numpy as np
from numpy import mean
from numba import njit
from numba.typed import List
from scipy.stats import skew

magnitude_type = list[float]
test_type      = list[list[float]]
templates_type = list[list[list[float]]]
sensor_type    = list[templates_type, test_type]
exercise_type  = list[sensor_type]
subject_type   = list[exercise_type]

def norm(A: test_type) -> magnitude_type:
    return [np.linalg.norm(vec) for vec in A]

@njit
def svd(A: test_type) -> test_type:
    """ Singular Value Decomposition
    
    Takes a M by N matrix A and transforms it using SVD.
    Then, using the Sigma and V transpose data as the orientation robust data, return this.
    
    Note, main data set returns data as A transpose. Meaning that rows and columns are transposed. However, for cotinuity, we return the 
    matrix to this switched version. Meaning that the result will be A (N X 3).
    
    """
    M, N = A.shape
    if M != 3:
        A = A.T
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    return (U.T @ A).T

def make_robust_absolute(A_I):
    return np.absolute(A_I)

def usvd_abs(A: test_type) -> test_type:
    M, N = A.shape
    if M != 3:
        A = A.T
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    robU = make_robust_absolute(U.T @ A)
    return robU.T

def make_robust_skewness(U, matrix):
    u1, u2, _ = U.T
    if skew(u1 @ matrix) < 0:
        u1 = -u1
    if skew(u2 @ matrix) < 0:
        u2 = -u2
    u3 = np.cross(u1, u2)
    return np.vstack((u1, u2, u3))
def usvd_skew(A: test_type) -> test_type:
    M, N = A.shape
    if M != 3:
        A = A.T
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    robU = make_robust_skewness(U, A)
    return (robU @ A).T

def make_robust_mean(U, matrix):
    u1, u2, _ = U.T
    if mean(u1 @ matrix) < 0:
        u1 = -u1
    if mean(u2 @ matrix) < 0:
        u2 = -u2
    u3 = np.cross(u1, u2)
    return np.vstack((u1, u2, u3))

def usvd_mean(A: test_type) -> test_type:
    M, N = A.shape
    if M != 3:
        A = A.T
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    robU = make_robust_mean(U, A)
    return (robU @ A).T

@njit
def slide_window(time_series: list[list[float]], window_size: int) -> list[list[float]]:
    """ Sliding Window
    
    Moves window over time series and performs SVD using the data in the window. Only the center point is stored. 
    Note: edge cases:
    1) Beginning: if the current time step is smaller than half the window size, we update the current point as if it is the center point. The window will not move.
    2) Ending: if the current time step + half of the window size is >= the length of the time series, the window will not move and the current point will become the center point.
    
    """
    results = List()
    max_step        = len(time_series)
    window_center   = window_size // 2
    start_of_window = 0
    end_of_window   = start_of_window + 2*window_center
    
    position = 0
    for time_step in range(max_step):
        if time_step < window_center or time_step + window_center >= max_step:
            position = time_step
            results.append((position, start_of_window, end_of_window))
        else:
            position = window_center
            results.append((position, start_of_window, end_of_window))
            start_of_window += 1
            end_of_window += 1
    return results

@njit
def wsvd(time_series, window_size):
    windows = slide_window(time_series, window_size)
    wsvd_time_series = np.zeros(time_series.shape)
    for idw, window in enumerate(windows):
        window_data = time_series[window[1]: window[2]]
        rel_pos     = window[0]
        if window[0] >= len(window_data):
            rel_pos = window_size - (window[2] - window[0] + 1)   
        wsvd_time_series[idw] = svd(window_data)[rel_pos]
    return wsvd_time_series

def wusvd_absolute(time_series, window_size):
    windows = slide_window(time_series, window_size)
    wusvd_time_series = np.zeros(time_series.shape)
    for idw, window in enumerate(windows):
        window_data = time_series[window[1]: window[2]]
        rel_pos     = window[0]
        if window[0] >= len(window_data):
            rel_pos = window_size - (window[2] - window[0] + 1)
        wusvd_time_series[idw] = usvd_abs(window_data)[rel_pos]
    return wusvd_time_series


def wusvd_skew(time_series, window_size):
    windows = slide_window(time_series, window_size)
    wusvd_time_series = np.zeros(time_series.shape)
    for idw, window in enumerate(windows):
        window_data = time_series[window[1]: window[2]]
        rel_pos     = window[0]
        if window[0] >= len(window_data):
            rel_pos = window_size - (window[2] - window[0] + 1)
        wusvd_time_series[idw] = usvd_skew(window_data)[rel_pos]
    return wusvd_time_series

def wusvd_mean(time_series, window_size):
    windows = slide_window(time_series, window_size)
    wsvd_time_series = np.zeros(time_series.shape)
    for idw, window in enumerate(windows):
        window_data = time_series[window[1]: window[2]]
        rel_pos     = window[0]
        if window[0] >= len(window_data):
            rel_pos = window_size - (window[2] - window[0] + 1)   
        wsvd_time_series[idw] = usvd_mean(window_data)[rel_pos]
    return wsvd_time_series

def simulate_rotation(time_series, degrees):
    radians = [degree * (np.pi / 180) for degree in degrees]
    matrix = time_series.copy()
    theta_rad = radians[0]
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(theta_rad), -np.sin(theta_rad)],
                      [0, np.sin(theta_rad), np.cos(theta_rad)]])

    phi_rad = radians[1]
    rot_y = np.array([[np.cos(phi_rad), 0, np.sin(phi_rad)],
                      [0, 1, 0],
                      [-np.sin(phi_rad), 0, np.cos(phi_rad)]])

    psi_rad = radians[2]
    rot_z = np.array([[np.cos(psi_rad), -np.sin(psi_rad), 0],
                      [np.sin(psi_rad), np.cos(psi_rad), 0],
                      [0, 0, 1]])
    rot_matrix = rot_z @ rot_y @ rot_x
    matrix = rot_matrix @ matrix.T
    return matrix.T

def simulate_sudden_changes(time_series, list_of_changes, list_of_angles):
    matrix = time_series.copy()
    for ((start, end), angles) in zip(list_of_changes, list_of_angles):
        matrix[start:end] = simulate_rotation(matrix[start:end], angles)
    return matrix