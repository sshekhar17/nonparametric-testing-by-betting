from time import time 
import numpy as np 
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm 


def truncateRows(X, radius=1):
    """
    Given a 2d array X, find the rows whose 
    norm is larger than radius, and scale those 
    rows to make their norm equal to radius 
    """
    assert radius>0 
    normX = np.linalg.norm(X, axis=1) 
    idx = np.where(normX>radius) 
    X[idx] = (X[idx]*radius)/(normX[idx].reshape((-1,1))) 
    return X 

def median_heuristic(Z):
    # compute the pairwise distance between the elements of Z 
    dists_ = pdist(Z)
    # obtain the median of these pairwise distances 
    sig = np.median(dists_)
    return sig


def RBFkernel(x, y=None, bw=1.0):
    y = x if y is None else y 
    dists = cdist(x, y, 'euclidean') 
    sq_dists = dists * dists 
    K = np.exp(-sq_dists/(2*bw*bw))
    return K 

def LinearKernel(x, y=None):
    y = x if y is None else y 
    K = np.einsum('ji, ki ->jk', x, y) 
    return K 

def PolynomialKernel(x, y=None, c=1.0, p=2):
    L = LinearKernel(x, y) 
    K = (c + L)**p 
    return K 

def permuteXY(X, Y, perm=None):
    Z = np.concatenate((X, Y), axis=0)
    nZ, nX = len(Z), len(X)
    if perm is None:
        perm = np.random.permutation(nZ) 
    idxX, idxY = perm[:nX], perm[nX:]
    X_, Y_ = Z[idxX], Z[idxY]
    return X_, Y_

def permutationTwoSampleTest(X, Y, statfunc, params=None, num_perms=200):
    params = {} if params is None else params 
    stat = statfunc(X, Y, **params)

    V = np.zeros((num_perms,))
    nZ = len(X) + len(Y)
    for i in range(num_perms):
        perm = np.random.permutation(nZ)
        X_, Y_ = permuteXY(X, Y, perm=perm) 
        val = statfunc(X_, Y_, **params)
        V[i] = val
    # compute the p-value 
    p = len(V[V>=stat])/num_perms
    return p 


def runBatchTwoSampleTest(Source, statfunc, params=None, num_perms=200,
                        alpha=0.05, Nmax=200, num_steps=20, initial=10, 
                        num_trials=200, progress_bar=False, store_times=False):
    # generate the different sample-sizes to be used in the power curve 
    NN = np.linspace(start=initial, stop=Nmax, num=num_steps, dtype=int) 
    # initialize the array to hold power values 
    Power = np.zeros(NN.shape) 
    # initialize the array to store average running times 
    if store_times:
        Times = np.zeros(NN.shape)

    range_ = range(num_trials) 
    range_ = tqdm(range_) if progress_bar else range_

    for trial in range_: 
        for i, n in enumerate(NN): 
            # do one trial of the test with sample-size equal to n 
            X, Y = Source(n) 
            t0 = time()
            p = permutationTwoSampleTest(X, Y, statfunc, num_perms=num_perms,
                                            params=params) 
            t1 = time() - t0
            # update the power value 
            if p<=alpha:
                Power[i] += 1 
            # update the running time 
            if store_times:
                Times[i] += t1 
    Power /= num_trials 
    if store_times:
        Times /= num_trials 
        return Power, Times, NN
    else:
        return Power, NN 


def get_power_from_stopping_times(StoppingTimes, N):
    num_trials = len(StoppingTimes) 
    S = StoppingTimes[StoppingTimes<N] 
    S = np.sort(S)
    Power = np.zeros((N,))
    for s in S: 
        Power[int(s-1):] += 1
    Power /= num_trials 
    return Power 