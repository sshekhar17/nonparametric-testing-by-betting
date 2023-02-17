from tqdm import tqdm 
from functools import partial
from math import sqrt 

import numpy as np 
from SeqTestsUtils import KellyBettingApprox, ONSstrategy, deLaPenaMartingale, runSequentialTest
from utils import RBFkernel, median_heuristic
from utils import runBatchTwoSampleTest
from sources import GaussianSource
import matplotlib.pyplot as plt 
import seaborn as sns


# a helper function 
def computeMMD(X, Y, kernel=None, perm=None, biased=True):
    """
    Compute the quadratic time MMD statistic based on gram-matrix K. 

    X       :ndarray    (nX, ndims) size observations
    Y       :ndarray    (nY, ndims) size observations
    kernel  :callable   kernel function 
    perm    :ndarray    the permutation array 
    biased  :bool       if True, compute the biased MMD statistic 

    returns 
    -------
    mmd     :float      the quadratic-time MMD statistic. 
    """
    Z = np.concatenate((X, Y), axis=0)
    nX, nZ = len(X), len(Z)
    # default kernel is RBF kernel 
    if kernel is None:
        bw = median_heuristic(Z)
        kernel = partial(RBFkernel, bw=bw)
    # # default value of perm is the indentity permutation
    if perm is None: 
        perm = np.arange(nZ)
    # obtain the X and Y indices 
    idxX, idxY = perm[:nX], perm[nX:]
    # permuted rows of observations 
    X_, Y_ = Z[idxX], Z[idxY]
    # extract the required matrices 
    KXX = kernel(X_, X_)
    KYY = kernel(Y_, Y_)
    KXY = kernel(X_, Y_)
    # compute the mmd statistic 
    nY = nZ - nX
    nY2, nX2, nXY = nY*nY, nX*nX, nX*nY
    assert nY>0 
    if biased:
        mmd = sqrt((1/nX2)*KXX.sum() + (1/nY2)*KYY.sum() - (2/nXY)*KXY.sum()) 
    else:#TODO: implement the unbiased mmd statistic 
        raise NotImplementedError
    # return the mmd statistic
    return mmd 


def kernelMMDprediction(X, Y, kernel=None, post_processing=None):
    nX, nY = len(X), len(Y) 
    assert nX==nY # only works with paired observations 
    assert nX>20
    # default kernel is RBF kernel 
    if kernel is None:
        # use the first 20 pairs of observations for bandwidth selection
        # TODO: get rid of this hardcoding and update the bandwidth 
        # after every block of observations 
        bw = median_heuristic(np.concatenate((X[:20], Y[:20]), axis=0))
        kernel = partial(RBFkernel, bw=bw)
    KXX = kernel(X, X)
    KYY = kernel(Y, Y)
    KXY = kernel(X, Y) 
    F = np.zeros((nX,)) 
    F_ = np.zeros((nX,))
    for i in range(1, nX):
        termX = np.mean((KXX[i, :i] - KXY[i, :i]))
        termY = np.mean((KXY[:i, i] - KYY[:i, i]))
        F_[i] = (termX - termY)
        F[i] = (termX - termY)
        ### a heuristic that significantly improve the 
        ### practical performance
        if i>10:
            i0 = max(0, i-50)
            max_val = np.max(F_[:i]) 
            F[i] =F_[i] / max_val 
    if post_processing=='sinh':
        F = np.sinh(F) 
    elif post_processing=='tanh':
        F = np.tanh(F)
    elif post_processing=='arctan':
        F = (2/np.pi)*np.arctan(F)  
    elif post_processing=='delapena':
        F = deLaPenaMartingale(F)
    return F 

