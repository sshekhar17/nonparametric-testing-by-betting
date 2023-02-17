"""
Implementation of 
    1.  the Sequential Two-sample kernel MMD test from 
        Balasubramani and Ramdas (2016). 
        Link to the paper: https://arxiv.org/pdf/1506.03486.pdf
    2. the sequential test in Sec 4.2 of Manole and Ramdas (2021)
        Link to the paper: https://arxiv.org/pdf/2103.09267.pdf 
"""
from time import time 
from math import log, sqrt, log2, pi 
import multiprocess as mp

import numpy as np 
from tqdm import tqdm 
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt 
from scipy.special import zeta as zeta_func
from sklearn.gaussian_process.kernels import RBF 

from constants import LIL_THRESHOLD_CONSTANT, MR_THRESHOLD_CONSTANT, ZETA_2
from utils import get_power_from_stopping_times

plt.style.use('seaborn-whitegrid')
# matplotlib.style.use('seaborn-white')
#===============================================================================
## Sequential Test from Balasubramani and Ramdas (2016)
#===============================================================================
def calculate_H(X, Y, kernel=None):
    """
    Compute and return the H-statistic from Balasubramani-Ramdas 2016

    X       --  (N, num_features) array containing N i.i.d. draws from P
    Y       --  (N, num_features) array containing N i.i.d. draws from Q
    kernel  --  a sklearn.gaussian_process.kernel object, if None it is 
                initialized to an RBF with lengthscale 1.0.

    Returns:
    H       -- (N//2,) numpy array containing the linear 
                time MMD statistic: 
                H[i] = K(x_{2i}, x_{2i+1}) + K(y_{2i}, y_{2i+1}) 
                      - K(x_{2i}, y_{2i+1}) - K(x_{2i+1}, y_{2i})
                for i=0, 1, \ldots (N//2-1)

    """
    # set kernel to default value if no kernel is passed
    kernel = RBF(length_scale=1.0) if (kernel is None) else kernel
    # define a helper function to compute h_i
    def helper(i):
        assert 2*i+1<len(X) 
        x1, x2 = X[2*i].reshape((1,-1)), X[2*i+1].reshape((1,-1)) 
        y1, y2 = Y[2*i].reshape((1,-1)), Y[2*i+1].reshape((1,-1)) 
        a = (kernel(x1, x2) + kernel(y1, y2) - kernel(x1, y2)
               - kernel(x2, y1))
        a = a.reshape((1,))[0]
        return a
    # compute the linear-time MMD statistic
    H = np.array([
        helper(i) for i in range( len(X) // 2 )
    ])
    return H 



def oneLilTest(X, Y, kernel=None, alpha=0.05):
    """
    Return the statistic and threshold for the LIL test of Balasubramani and Ramdas 2016.

    X       --  (N, num_features) array containing N i.i.d. draws from P
    Y       --  (N, num_features) array containing N i.i.d. draws from Q
    kernel  --  a sklearn.gaussian_process.kernel object, if None it is 
                initialized to an RBF with lengthscale 1.0.
    alpha   --  float \in (0,1): type-I error bound 

    Returns:
    T       -- statistics -- numpy array (N//2, ) consisting of 
                the test statistic
    Q       -- thresholds -- numpy array (N//2, ) according to the 
                expression derived in Balasubramani-Ramdas (2016)
                using empirical Bernstein LIL concentration result.
    """
    # compute H
    H = calculate_H(X=X, Y=Y, kernel=kernel)
    # statistic is the cumulative sum of H
    T = np.cumsum(H)
    # compute the empirical variance
    V_hat = np.cumsum(H**2) 
    # obtain the expression 
    Q = LIL_THRESHOLD_CONSTANT*(
        log(1/alpha) + np.sqrt(2*V_hat*np.log( np.log(np.maximum(2 , V_hat)) /alpha ))
    )
    return T, Q

def compute_stopping_time(
    stats,
    thresholds, 
    factor=2
):
    """
    Return the stopping times from test statistics and thresholds
    """
    assert stats.shape==thresholds.shape
    num_trials, N_over_2 = stats.shape
    StoppingTimes = np.zeros((num_trials,))
    for i, (st, th) in enumerate(zip(stats, thresholds)):
        # index st crosses th
        t = np.argmax((st-th)>=0)
        # t=0 indicates that threshold not crossed
        StoppingTimes[i] = factor*N_over_2+1 if t==0 else factor*t+1
        # multiplication by factor (==2): to reflect the actual sample size 
    return StoppingTimes


def runLilTest(Source, N=500, num_trials=100, kernel=None, alpha=0.05,
            parallel=False,  progress_bar=True, return_type=1):
    """
    Run several trials of LIL test of Balsubramani-Ramdas (2016),
    return the stopping times, Power and TypeI error 
    """
    # assign default value if kernel is None
    kernel = RBF(length_scale=1.0) if (kernel is None) else kernel
    # initialize
    stats, th = np.zeros((num_trials, N//2)), np.zeros((num_trials, N//2))

    range_ = tqdm(range(num_trials)) if progress_bar else range(num_trials)
    for i in range_:
        X, Y = Source(N)
        T, Q = oneLilTest(X=X, Y=Y, kernel=kernel, alpha=alpha)
        stats[i], th[i] = T, Q
    # obtain the stopping times, power 
    StoppingTimes = compute_stopping_time(stats, th, factor=2.0)
    Power = np.array([ (StoppingTimes<=i+1).sum() for i in range(N)])/num_trials

    if return_type==1:
        return StoppingTimes, Power
    else:
        return stats, th
#===============================================================================
## Sequential Test from Manole and Ramdas (2021)
#===============================================================================
def compute_thresholdMR(t, B=1.0, alpha=0.05, tol=1e-10): 
    # define the helper function 
    t = max(t, 2) # ensure that t>2
    def g_func(a):
        assert a>0
        a = max(1, a) 
        return  (a**2)*ZETA_2
    term1 = 4*sqrt(2*B/t) 
    term2 = max( log(max(2*g_func(log2(t)), tol)) + log(2/alpha), tol)
    return MR_THRESHOLD_CONSTANT*term1*(1+sqrt(term2))


def oneMRTest(X, Y, kernel=None, B=1.0, alpha=0.05):
    """
    return the power and type-I error of MR sequential MMD test 

    X       --  (N, num_features) array containing N i.i.d. draws from P
    Y       --  (N, num_features) array containing N i.i.d. draws from Q
    kernel  --  a sklearn.gaussian_process.kernel object, if None it is 
                initialized to an RBF with lengthscale 1.0.
    alpha   --  float \in (0,1): type-I error bound 

    Returns:
    T       -- statistics -- numpy array (N, ) consisting of 
                the biased quadratic time MMD test statistic
    Q       -- thresholds -- numpy array (N, ) according to the 
                expression derived in  Manole and Ramdas (2021)
    """
    kernel = RBF(length_scale=1.0) if kernel is None else kernel     
    N = len(X)
    T, Q = np.zeros((N,)), np.zeros((N,))
    if len(X)<2:
        raise Exception("Enter X and Y with more than 2 rows")
    # define a helper function
    def helper(i, A, B=None):
        B = A if B is None else B 
        a, b = A[i].reshape((1,-1)), B[i].reshape((1,-1))

        return_val = (kernel(A[:i], b).sum()  
                      + kernel(B[:i], a).sum()  
                      + kernel(a, b).sum())
        return return_val 
    kXX, kYY = kernel(X[:2], X[:2]).sum(), kernel(Y[:2], Y[:2]).sum()
    kXY = kernel(X[:2], Y[:2]).sum()
    T[1] = (1/2)*sqrt(kXX + kYY - 2*kXY)  
    for i in range(2, N):
        kXX += helper(i, A=X)
        kYY += helper(i, A=Y)
        kXY += helper(i, A=X, B=Y)
        temp = kXX + kYY  - 2*kXY
        if temp<0:
            print('MMD**2<0???')
            temp=0
        T[i] = (1/(i+1))*sqrt(temp) 
    # now compute the threshold 
    Q = np.array(
        [compute_thresholdMR(t, B=B, alpha=alpha) for t in range(1, N+1)]
        ) 
    return T, Q 


def runMRTest(Source,  N=500, num_trials=100, kernel=None,
        B=1.0, alpha=0.05, parallel=False, progress_bar=True,
        return_type=1):
    """
    Run several trials of MR test, return the stopping times
    """
    # assign default value if kernel is None
    kernel = RBF(length_scale=1.0) if (kernel is None) else kernel
    # initialize
    stats, th = np.zeros((num_trials, N)), np.zeros((num_trials, N))
    # run the main experiment
    _range = tqdm(range(num_trials)) if progress_bar else range(num_trials)
    for i in _range:
        X, Y = Source(N)
        T, Q = oneMRTest(X=X, Y=Y, kernel=kernel, B=B, alpha=alpha)
        stats[i], th[i]= T, Q
    # obtain the stopping times, power and typeI error 
    StoppingTimes = compute_stopping_time(stats, th, factor=1.0)
    Power = np.array([ (StoppingTimes<=i+1).sum() for i in range(N)])/num_trials
    if return_type==1:
        return StoppingTimes, Power
    else:
        return stats, th
#===============================================================================
#===============================================================================
def DarlingRobbinsThreshold(t, a=2, m=10, alpha=0.05, one_sample=False):
    assert m>1
    assert t>0
    b = (1/(alpha*(m-1)))
    factor= 1.0 if one_sample else 2.0
    b = 4*sqrt(2)*b  if not one_sample else b 
    return factor*sqrt((t+1)*(a*log(t)+ log(b)))/t

def HowardRamdasDKWthreshold(t, m=10, one_sample=False, alpha=0.05): 
    if one_sample:
        C = max(7, 0.8*log( 1612/alpha)) 
        threshold =  0.85*sqrt(  (log(1 + max(0, log( t/m))) + C)/t )
    else:
        C = max(7, 0.8*log( 1612*2/alpha)) 
        threshold =  2*0.85*sqrt(  (log(1 + log( t/m)) + C)/t )
    return threshold 

def ManoleRamdasDKWthreshold(t, one_sample=False, alpha=0.05):
    if one_sample:
        # Corollary 12 of Manole Ramdas
        log_ell = log( ((max(1, log2(t)))**2)*ZETA_2 )
        threshold =  sqrt(pi/t) + 2*sqrt( (2/t)*(log_ell + log(1/alpha)) )
    else:
        # Corollary 13 of Manole Ramdas
        g = np.exp(1)*(max(2, 2*log2(t))**3)*(zeta_func(2)-zeta_func(3))
        log_g = log(g)
        threshold =  2*sqrt(pi/t) + 2*sqrt((1/t)*(log_g + log(1/alpha)))
    return  threshold


def runSeqDKWTest(Source, N=500, num_trials=100, th_func=None,
        th_args=None, th_kwargs=None, alpha=0.05, parallel=False, 
        progress_bar=True, one_sample=False, min_len=5, test_name='DR'): 


    if th_func is None:
        if test_name=='DR':
            th_func = DarlingRobbinsThreshold
        elif test_name=='MR':
            th_func = ManoleRamdasDKWthreshold
        elif test_name=='HR':
            th_func=HowardRamdasDKWthreshold

    th_args = () if th_args is None else th_args 
    th_kwargs =({'one_sample':one_sample, 'alpha':alpha} if 
                    th_kwargs is None else th_kwargs)
    
    if not one_sample: 
        stat_func = stats.ks_2samp
    else:
        raise Exception('Not implemented one-sample KS test')

    def helper(i):
        X, Y = Source(N)
        rejected = False 
        stoppingTime = N
        for t in range(min_len, N):
            stat, _ = stat_func(X[:t], Y[:t])
            th = th_func(t, *th_args, **th_kwargs)

            if not rejected and stat>th:
                rejected = True 
                stoppingTime = t+1
        return rejected, stoppingTime

    range_ = range(num_trials)
    range_ = tqdm(range_) if progress_bar else range_ 

    StoppingTimes = np.zeros((num_trials,))
    for trial in range_: 
        _, stoppingTime = helper(trial) 
        StoppingTimes[trial] = stoppingTime
    Power = get_power_from_stopping_times(StoppingTimes, N)

    return Power, StoppingTimes
    