"""
Experiment 2 from Section VI: "Numerical Experiments"
Adaptivity of the sequential test to the problem hardness 
"""

import argparse
import pickle
from functools import partial
from math import sqrt
from time import time 
from tqdm import tqdm 

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from utils import RBFkernel, runBatchTwoSampleTest
from sources import GaussianSource, getGaussianSourceparams

from kernelMMD import computeMMD, kernelMMDprediction
from SeqTestsUtils import runSequentialTest, ONSstrategy, KellyBettingApprox

def main(N_batch=800, N_betting=2400, d=10, num_perms=100,num_trials=200,
            alpha=0.05, epsilon_mean=0.35, epsilon_var=1.5, num_pert_mean=1, 
            num_pert_var=0, num_steps_batch=10, progress_bar=False):
   
    meanX, meanY, covX, covY = getGaussianSourceparams(d=d, epsilon_mean=epsilon_mean, 
                                    epsilon_var=epsilon_var,
                                    num_perturbations_mean=num_pert_mean, 
                                    num_perturbations_var=num_pert_var)
    # generate the source 
    Source = GaussianSource(meanX=meanX, meanY=meanY, 
                            covX=covX, covY=covY, truncated=False)

    # Initialize the kernel 
    # set the bandwidth according to the dimension
    kernel = partial(RBFkernel, bw=sqrt(d/2))
    ####=========================================================
    # Do the batch mmd test 
    statfunc = partial(computeMMD, kernel=kernel)
    PowerBatch, TimesBatch, NNBatch = runBatchTwoSampleTest(Source, statfunc, 
                                num_perms=num_perms, progress_bar=progress_bar, num_steps=num_steps_batch,
                                num_trials=num_trials, Nmax=N_batch, store_times=True)
    # deltaT = time() - t0 
    # print(f'Batch test took {deltaT:.2f} seconds')
    ####=========================================================
    # Do the betting based sequential kernel-MMD test 
    # print(f'\n====Starting Betting Sequential test====\n')
    # t0 = time()
    Prediction = kernelMMDprediction
    Betting = ONSstrategy
    # Betting = KellyBettingApprox
    pred_params=None 
    bet_params=None
    PowerBetting, StoppedBetting, StoppingTimesBetting = runSequentialTest(Source, Prediction, Betting, 
                                                            alpha=alpha, Nmax=N_betting,
                                                            pred_params=pred_params, bet_params=bet_params,
                                                            num_trials=num_trials, progress_bar=progress_bar)
    mean_stopping_time_betting = StoppingTimesBetting.mean()
    NNBetting = np.arange(1, N_betting+1)
    # deltaT = time() - t0 
    # print(f'Sequential Betting test took {deltaT:.2f} seconds')

    return PowerBatch, NNBatch, mean_stopping_time_betting

####=========================================================
#### PLot the results 
####=========================================================
def plot_results(Data, title, xlabel, ylabel, savefig=False, figname=None):
    palette = sns.color_palette(n_colors=10)
    plt.figure() 
    i=0
    for epsilon in Data:
        powerBatch, nBatch, meanTau = Data[epsilon] 
        plt.plot(nBatch, powerBatch, label=str(epsilon), color=palette[i])
        plt.axvline(x=meanTau, linestyle='--', color=palette[i])
        plt.xlabel(xlabel, fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        plt.title(title, fontsize=15)
        i+=1 
    plt.legend(fontsize=12)
    if savefig:
        figname = 'temp.png' if figname is None else figname 
        plt.savefig(figname, dpi=450)
    else:
        plt.show()

if __name__=='__main__':
 # Maximum sample size 
    Epsilon_mean = np.flip(np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.7]))
    NN_batch = np.flip(np.array([1200, 1000, 600, 500, 450, 250]))
    NN_betting = 3*NN_batch 

    d=10
    num_perms = 200
    num_trials= 500
    alpha=0.05
    num_steps_batch= 20
    ### parameters of the Gaussian distribution 
    epsilon_var = 1.5 
    num_pert_mean=1 
    num_pert_var = 0

    ######################
    savefig=True
    savedata=savefig
    ######################

    DataToPlot = {}
    for i, epsilon_mean in tqdm(list(enumerate(Epsilon_mean))):
        print(f'Starting the GMD experiment with (d, epsilon):({d}, {epsilon_mean})')
        N_batch, N_betting = NN_batch[i], NN_betting[i] 
        powerBatch, samplesizeBatch, meanTau = main(N_batch=N_batch, N_betting=N_betting, d=d,
                    num_perms=num_perms, num_trials=num_trials,
                    alpha=alpha, epsilon_mean=epsilon_mean, 
                    epsilon_var=epsilon_var, num_pert_mean=num_pert_mean, 
                    num_pert_var=num_pert_var,
                    num_steps_batch=num_steps_batch, progress_bar=False)
        DataToPlot[epsilon_mean] = (powerBatch, samplesizeBatch, meanTau)
        print('__'*20, '\n')

    title='Adaptivity of Sequential Test to Alternative'
    xlabel='Sample-Size (n)'
    ylabel='Power'
    figname = './data/Adaptivity1.png'
    plot_results(DataToPlot, title, xlabel, ylabel, savefig=savefig, figname=figname)

    filename = './data/Experiment2data.pkl'
    with open(filename, 'wb') as handle: 
        pickle.dump(DataToPlot, handle)