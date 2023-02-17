"""
Classes generating different data sources
"""
import numpy as np
import scipy.stats as stats 
from utils import truncateRows

def getGaussianSourceparams(d=10, epsilon_mean=0.5, epsilon_var=0.0,
                            num_perturbations_mean=2,
                            num_perturbations_var=2): 
    meanX, meanY = np.zeros((d,)), np.zeros((d,))
    meanY[:num_perturbations_mean] = epsilon_mean 
    # get the covariance matrices 
    covX = np.eye(d)
    diagY = np.ones((d,)) 
    diagY[:num_perturbations_var] = epsilon_var 
    covY = np.diag(diagY) 
    return meanX, meanY, covX, covY
    


def GaussianSource(meanX=None, meanY=None, covX=None, covY=None,
                truncated=False, radius=None, epsilon=0.5):
    if meanX is None: # set all params to default
        d=5
        meanX = np.zeros((d,)) 
        meanY = np.ones((d,))*epsilon
        covX = np.eye(d)
        covY = np.eye(d)
    
    if truncated:
        radius = 1 if radius is None else radius 
        assert radius > 0 
    def Source(n, m=None, truncated=truncated, radius=radius):
        m = n if m is None else m
        X = stats.multivariate_normal.rvs(mean=meanX, cov=covX, size=n)
        Y = stats.multivariate_normal.rvs(mean=meanY, cov=covY, size=m)
        if truncated:
            X = truncateRows(X, radius=radius)
            Y = truncateRows(Y, radius=radius)
        return X, Y
    return Source 


def TdistSource(df1=1, df2=1, scale1=1.0, scale2=1.0, 
                loc1=0.0, loc2=0.0):

    def Source(n, m=None):
        m = n if m is None else m 
        X = stats.t.rvs(size=n, loc=loc1, df=df1, scale=scale1)
        Y = stats.t.rvs(size=n, loc=loc2, df=df2, scale=scale2)
        return X, Y 
    return Source 

def SourceFromDataset():
    pass 


