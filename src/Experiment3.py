"""
Experiment 3 from Section VI: "Numerical Experiments"
Adaptivity of the sequential test with unbounded kernels to the problem hardness 
"""
import os
import pickle 
import argparse
from functools import partial
from math import sqrt
from time import time 

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from utils import LinearKernel, PolynomialKernel, RBFkernel, runBatchTwoSampleTest
from sources import GaussianSource, getGaussianSourceparams

from kernelMMD import computeMMD, kernelMMDprediction
from SeqTestsUtils import runSequentialTest, ONSstrategy, KellyBettingApprox



def main(Source, N_batch=200, N_betting=600,d=10, num_perms=10,num_trials=20,
            alpha=0.05, num_steps_batch=10, progress_bar=False):
   
    # Maximum sample size 
    N_MR = 2*N_betting
    N_BR = 2*N_betting

    # Initialize the kernel 
    # kernel = LinearKernel
    kernel = PolynomialKernel
    ####=========================================================
    # Do the batch mmd test 
    # print(f'\n====Starting Batch MMD Test with {num_perms} permutations====\n')
    t0 = time()
    statfunc = partial(computeMMD, kernel=kernel)
    PowerBatch, NNBatch = runBatchTwoSampleTest(Source, statfunc, 
                                num_perms=num_perms, progress_bar=progress_bar, 
                                num_trials=num_trials, Nmax=N_batch, 
                                num_steps=num_steps_batch)
    deltaT = time() - t0 
    # print(f'Batch test took {deltaT:.2f} seconds')
    ####=========================================================
   
    ####=========================================================
    # Do the symmetry test based  sequential kernel-MMD test 
    # print(f'\n====Starting Symmetry Sequential test using Tanh====\n')
    t0 = time()
    PredictionTanh = partial(kernelMMDprediction, post_processing='tanh')
    pred_params=None 
    bet_params=None
    Betting2 = KellyBettingApprox
    Betting = ONSstrategy
    PowerTanh, StoppedTanh, StoppingTimesTanh = runSequentialTest(Source, PredictionTanh, Betting2, 
                                                            alpha=alpha, Nmax=N_betting,
                                                            pred_params=pred_params, bet_params=bet_params,
                                                            num_trials=num_trials, progress_bar=True)
    mean_tau_tanh = StoppingTimesTanh.mean()
    NNTanh = np.arange(1, N_betting+1)
    deltaT = time() - t0 
    # print(f'Sequential Tanh based two-sample test test took {deltaT:.2f} seconds')

    ####=========================================================
    
    ## Prepare the data for plotting 
    # Data = {}
    # Data['batch'] = (PowerBatch, NNBatch) 
    # Data['betting+tanh']=(PowerTanh, NNTanh, mean_tau_tanh, StoppingTimesTanh) 
    # return Data 
    return PowerBatch, NNBatch, mean_tau_tanh


def plot_results(Data, title, xlabel, ylabel, savefig=False, figname=None):
    palette = sns.color_palette(n_colors=15)
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
    parser = argparse.ArgumentParser() 
    parser.add_argument('--d', default=10, type=int, help='data dimension')
    parser.add_argument('--eps', default=0.5, type=float)
    parser.add_argument('--num_perms', '-np', default=200, type=int)
    parser.add_argument('--num_trials', '-nt', default=500, type=int)
    parser.add_argument('--save_fig', '-sf', action='store_true')
    parser.add_argument('--save_data', '-sd', action='store_true')
    parser.add_argument('--alpha', '-a', default=0.05, type=float)
    args = parser.parse_args()


    d=args.d
    Epsilon_mean = np.flip(np.array([0.55, 0.60, 0.8, 1.0, 1.2, 2.0]))
    NN_batch = np.flip(np.array([900, 800, 600, 400, 300, 200]))
    NN_betting = 2*NN_batch 
    num_perms = args.num_perms
    num_trials=args.num_trials
    alpha=args.alpha
    num_steps_batch= 20
    ### parameters of the Gaussian distribution 
    num_pert_mean=1 # number of perturbations in the mean vector 
    epsilon_mean = args.eps # magnitude of the perturbation

    #====================
    savefig=args.save_fig
    savedata=args.save_data
    #====================
    DataToPlot = {}
    t0 = time()
    for i, epsilon_mean in enumerate(Epsilon_mean):
        print(f'\n Starting {i+1}/{len(Epsilon_mean)} interation with epsilon = {epsilon_mean}\n')
        N_batch, N_betting = NN_batch[i], NN_betting[i]
        meanX, meanY, covX, covY = getGaussianSourceparams(d=d, epsilon_mean=epsilon_mean, 
                                        epsilon_var=0,
                                        num_perturbations_mean=num_pert_mean, 
                                        num_perturbations_var=0)
        # increase the covariance of the observations to make them 
        # more spread out in the space. 
        covX *= 5 
        covY *= 5 
        # generate the source 
        Source = GaussianSource(meanX=meanX, meanY=meanY, 
                                covX=covX, covY=covY, truncated=False)

        # run the experiment
        data_ = main(Source, N_batch=N_batch, N_betting=N_betting, 
                        num_perms=num_perms,num_trials=num_trials, d=d,
                        alpha=alpha, num_steps_batch=num_steps_batch,
                        progress_bar=False)

        DataToPlot[epsilon_mean] = data_  

        print(f'Completed iteration {i+1}/{len(Epsilon_mean)}. Total Time elapsed: {time()-t0:.2f} seconds')

    # get the path of the file to store data 
    parent_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    data_dir = parent_dir + '/data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    temp = 'AdaptivityUnbounded'
    figname = f'{data_dir}/{temp}.png'
    # Create the figure
    title = 'Adaptivity to the unknown alternative'
    xlabel='Sample-Size'
    ylabel='Power'
    plot_results(DataToPlot, title=title, xlabel=xlabel, ylabel=ylabel, savefig=savefig,
                figname=figname)

    filename = f'{data_dir}/{temp}.pkl'
    if savedata:
        with open(filename, 'wb') as handle: 
            pickle.dump(DataToPlot, handle)

