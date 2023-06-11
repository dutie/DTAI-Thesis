"""
thesis.subsequence_dtw
~~~~~~~~~~~~~~~~~~~~~~

Performs subsequence search using DTW.
Based on the code in Müller, FMP, Springer 2015.

:author Mats Fockaert (Based on Müller, FMP, Springer 2015)
:copyright: Copyright 2023 KU Leuven
:license: Apache License, Version 2.0, see LICENSE for details.

"""


import numpy as np
from numba import njit

from scipy.spatial import distance

magnitude_type  = list[float]
test_type       = list[list[float]]
templates_type  = list[list[list[float]]]
sensor_type     = list[templates_type, test_type]
exercise_type   = list[sensor_type]
subject_type    = list[exercise_type]
costmatrix_type = list[list[float]] 

def mag_compute_cost_matrix(template: magnitude_type, test: magnitude_type) -> costmatrix_type:
    """
    Cost Matrix.
    
    Function that calculates the distance between each entry in a longer sequence (`test`) and a subsequence (`template`) and stores this in a matrix.
    
    :param template: Subsequence that we want to find.
    :param test: Sequence that contains (possibly) many occurences of template (-like) series.
    
    Returns: Cost matrix as a `costmatrix_type` object.
    """
    costMatrix = np.zeros((len(template), len(test)))
    for i in range(len(template)):
        for j in range(len(test)):
            costMatrix[i, j] = abs(template[i]- test[j])
    return costMatrix

def compute_cost_matrix(template: test_type, test: test_type) -> costmatrix_type:
    """
    Cost Matrix.
    
    Function that calculates the distance between each entry in a longer sequence (`test`) and a subsequence (`template`) and stores this in a matrix.
    
    :param template: Subsequence that we want to find.
    :param test: Sequence that contains (possibly) many occurences of template (-like) series.
    
    Returns: Cost matrix as a `costmatrix_type` object.
    """
    if np.ndim(test) > 1:
        X, Y = np.atleast_2d(template,test)
        costMatrix = distance.cdist(X,Y, metric ='euclidean')
    else:
        costMatrix = mag_compute_cost_matrix(template, test)
    return costMatrix

@njit
def warpingPath(acc_costMatrix: costmatrix_type, m=-1) -> tuple[costmatrix_type, int, int]:
    """
    The best warping path (or path of lowest accumulated cost) is built/found.
    
    :param acc_costMatrix: see :meth:`subsequence_dtw`
    
    Returns: The accumulated cost and the start, and end position of the warping path.
    """
    N : int = acc_costMatrix.shape[0]
    M : int = acc_costMatrix.shape[1]
    n = N - 1
    if m < 0:
        m = acc_costMatrix[N - 1, :].argmin()
    P = [(n,m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(acc_costMatrix[n-1, m-1], acc_costMatrix[n-1, m], acc_costMatrix[n, m-1])
            if val == acc_costMatrix[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == acc_costMatrix[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    start = P[0, 1]
    end = P[-1, 1] + 1
    acc_cost = acc_costMatrix[-1, end-1]
    return acc_cost, start, end

@njit  
def subsequence_dtw(costMatrix: costmatrix_type) -> tuple[costmatrix_type, int, int]:
    """
    Subsequence alignment using DTW.
    
    Find where the query (template) occurs in the longer series (test) using the cost matrix.
    
    :meth:`warping_path` and this method are based on Fundamentals of Music Processing, Meinard Müller, Springer, 2015.
    
    Uses dynamic programming to compute:
    accumulated cost at (n,m) = current cost + minimum of:
    - expansion: accumulated[n-1, m]
    - deletion: accumulated[n, m-1]
    - match: accumulated[n-1, m-1]
    
    Then, the lowest row 
    
    :param costMatrx: Matrix containing euclidean distance between entries in the template and in the longer series. 
    """    
    N : int = costMatrix.shape[0]
    M : int = costMatrix.shape[1]
    accumulated = np.zeros((N,M))
    accumulated[:, 0] = np.cumsum(costMatrix[:,0])
    accumulated[0, :] = costMatrix[0, :]
    for n in range(1,N):
        for m in range(1,M):
            accumulated[n,m] = costMatrix[n,m] + min(accumulated[n-1,m-1],
                                                     accumulated[n-1,m],
                                                     accumulated[n,m-1])
    b_star   = accumulated[-1,:].argmin()
    acc_cost = accumulated[-1, b_star]

    acc_cost, start, end = warpingPath(accumulated)
    return acc_cost, start, end