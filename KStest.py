from tqdm import tqdm 
from functools import partial
from math import sqrt 

import numpy as np 
import scipy.stats as stats
from SeqTestsUtils import KellyBettingApprox, ONSstrategy,  runSequentialTest
from SeqOther import runSeqDKWTest
from utils import RBFkernel, median_heuristic
from utils import runBatchTwoSampleTest
from sources import GaussianSource, TdistSource
import matplotlib.pyplot as plt 
import seaborn as sns


def KSprediction(X, Y, direction=0):
    nX = len(X)
    F = np.zeros((nX,))
    for i in range(1, nX):
        Xi, Yi = X[:i], Y[:i]
        # get the grid points 
        G = np.concatenate((Xi, Yi))
        # compute the empirical cdf at points in G 
        FXi = np.array([np.sum(Xi<=g) for g in G])
        FYi = np.array([np.sum(Yi<=g) for g in G])
        if direction==0:
            diffF = FXi - FYi 
        else: 
            diffF = FYi - FXi 
        idx = np.argmax(diffF) 
        u = G[idx] 
        if direction==0:
            F[i] = (X[i]<=u)*1.0 - (Y[i]<=u)*1.0
        else:
            F[i] = (Y[i]<=u)*1.0 - (X[i]<=u)*1.0
    return F 

def getKSstatistic2samp(X, Y):
    stat, _ = stats.ks_2samp(X, Y) 
    return stat

def BatchKS2samp(Source, Nmax, n_steps=20, initial=10, num_trials=100, 
                    alpha=0.05, progress_bar=False):
    initial = min(initial, Nmax//2)
    NNBatch = np.linspace(initial, Nmax, n_steps, dtype=int)
    PowerBatch = np.zeros(NNBatch.shape)
    range_ = range(num_trials)
    range_ = tqdm(range_) if progress_bar else range_
    for trial in range_: 
        for i, n in enumerate(NNBatch): 
            X, Y = Source(n) 
            _, pval = stats.ks_2samp(X, Y, alternative='two-sided') 
            PowerBatch[i] += 1.0*(pval<=alpha)
    PowerBatch /= num_trials 
    return PowerBatch, NNBatch 

