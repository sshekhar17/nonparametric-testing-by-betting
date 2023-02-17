from tqdm import tqdm 
from functools import partial
import numpy as np 
import scipy.stats as stats 
from math import log
import matplotlib.pyplot as plt 
from sources import GaussianSource



def ONSstrategy(F, lambda_max=0.5):
    """
    Compute the bets corresponding to the observations F 

    Parameters:
        F           :numpy array   of size (N,)
        lambda_max  :float positive real number in range (0,1)
    Returns:
        Lambda  :numpy array of size (N,) containing the bets
    
    Note that we use the wealth update rule:
    K_{t} = K_{t-1} \times (1 + \lambda_t F_t), 
    with a "+" instead of "-" used by Cutkosky & Orabona (2018). 
    """
    N = len(F)
    assert N>2
    Lambda = np.zeros((N,))
    A = 1
    c = 2/(2- log(3))
    for i in range(N-1):
        #update the z term 
        z = -F[i] / (1 + Lambda[i]*F[i])
        # update the A term 
        A += z**2 
        # get the new betting fraction 
        Lambda[i+1] = max(min(
            Lambda[i] - c * z / A, lambda_max
         ), -lambda_max)
    return Lambda 

def KellyBettingApprox(F, lambda_max=0.8):
    N = len(F) 
    assert N>2
    Lambda = np.zeros((N,))
    F2 = F*F 
    for i in range(1, N):
        lambda_ = F[:i].sum() / (F2[:i].sum() + 1e-10) 
        lambda_ = min(lambda_max, max(-lambda_max, lambda_)) 
        Lambda[i] = lambda_
    return Lambda



def get_stopping_time_from_wealth(W, alpha=0.05):
    th = 1/alpha 
    idx = np.where(W>=th)[0] 
    if len(idx)==0: # no stopping 
        stopped = False 
        stopping_time = len(W) 
    else:
        stopped = True 
        stopping_time = idx[0]+1  
    return stopped, stopping_time 


def runSequentialTest(Source, Prediction, Betting, alpha=0.05,
                            pred_params=None, bet_params=None, 
                            Nmax=1000, num_trials=50, progress_bar=False, 
                            hedge=False, hedge_weights=None, 
                            return_wealth=False):

    Power = np.zeros((Nmax,)) 
    StoppingTimes = np.zeros((num_trials,))
    Stopped = np.zeros((num_trials,))
    pred_params = {} if pred_params is None else pred_params
    bet_params = {} if bet_params is None else bet_params

    range_ = range(num_trials)
    range_ = tqdm(range_) if progress_bar else range_

    for trial in range_:
        X, Y = Source(Nmax) 
        # get the wealth process 
        if not hedge: # no hedgeing over different prediction strategies
            # get the payoff values 
            F = Prediction(X, Y, **pred_params)
            # get the betting fractions 
            Lambda = Betting(F, **bet_params) 
            W = np.cumprod(1 + Lambda*F) 
        else: #hedge over different prediction strategies 
            # some sanity checking 
            assert isinstance(Prediction, list) 
            assert isinstance(Betting, list) 
            nP = len(Prediction)
            assert nP==len(Betting)  
            if hedge_weights is None:
                # default weights are uniform 
                hedge_weights = np.ones((nP,)) 
            else:
                assert len(hedge_weights)==nP
            hedge_weights /= hedge_weights.sum()
            for j in range(nP):
                Fj = Prediction[j](X, Y, **pred_params)
                Lambdaj = Betting[j](Fj, **bet_params)
                if j==0:
                    W = hedge_weights[j]*np.cumprod(1 + Lambdaj*Fj) 
                else: 
                    W += hedge_weights[j]*np.cumprod(1 + Lambdaj*Fj) 
        # get the stopping_time 
        stopped, stopping_time = get_stopping_time_from_wealth(W, alpha)
        # update the results 
        Stopped[trial] = stopped 
        StoppingTimes[trial] = stopping_time 
        if stopped: 
            Power[stopping_time:] += 1
    Power /= num_trials 
    if return_wealth:
        return Power, Stopped, StoppingTimes, W 
    else:
        return Power, Stopped, StoppingTimes 


def deLaPenaMartingale(F):
    f_plus = np.exp(F - F*F/2) 
    f_minus = np.exp(-F - F*F/2) 
    idx1 = np.where(f_plus>=f_minus)[0]
    idx2 = np.where(f_plus<f_minus)[0] 
    f= np.zeros(F.shape) 
    f[idx1] = f_plus[idx1]
    f[idx2] = 2 - f_minus[idx2] 
    F_new = f - 1 
    return F_new 


