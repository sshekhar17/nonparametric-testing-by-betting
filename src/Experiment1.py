"""
Experiment 1 from Section VI: "Numerical Experiments"
Comparison of power of the sequential tests 
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

from utils import RBFkernel, runBatchTwoSampleTest
from sources import GaussianSource, getGaussianSourceparams

from kernelMMD import computeMMD, kernelMMDprediction
from SeqTestsUtils import runSequentialTest, ONSstrategy, KellyBettingApprox
from SeqLC import runLCexperiment 
from SeqOther import runLilTest, runMRTest


def main(N_batch=200, N_betting=600, d=10, num_perms=10,num_trials=20,
            alpha=0.05, epsilon_mean=0.35, epsilon_var=1.5, num_pert_mean=1, 
            num_pert_var=0, num_steps_batch=10, progress_bar=False):
   
    # Maximum sample size 
    N_MR = 2*N_betting
    N_BR = 2*N_betting
    N_LC = 2*N_betting  
    meanX, meanY, covX, covY = getGaussianSourceparams(d=d, epsilon_mean=epsilon_mean, 
                                    epsilon_var=epsilon_var,
                                    num_perturbations_mean=num_pert_mean, 
                                    num_perturbations_var=num_pert_var)
    # generate the source 
    Source = GaussianSource(meanX=meanX, meanY=meanY, 
                            covX=covX, covY=covY, truncated=False)

    # Initialize the kernel 
    kernel = partial(RBFkernel, bw=sqrt(d/2))
    ####=========================================================
    # Do the batch mmd test 
    print(f'\n====Starting Batch MMD Test with {num_perms} permutations====\n')
    t0 = time()
    statfunc = partial(computeMMD, kernel=kernel)
    PowerBatch, NNBatch = runBatchTwoSampleTest(Source, statfunc, 
                                num_perms=num_perms, progress_bar=progress_bar, 
                                num_trials=num_trials, Nmax=N_batch, 
                                num_steps=num_steps_batch)
    deltaT = time() - t0 
    print(f'Batch test took {deltaT:.2f} seconds')
    ####=========================================================
    # Do the betting based sequential kernel-MMD test 
    print(f'\n====Starting Betting Sequential test====\n')
    t0 = time()
    Prediction = kernelMMDprediction
    Betting = ONSstrategy
    pred_params=None 
    bet_params=None
    PowerBetting, StoppedBetting, StoppingTimesBetting = runSequentialTest(Source, Prediction, Betting, 
                                                            alpha=alpha, Nmax=N_betting,
                                                            pred_params=pred_params, bet_params=bet_params,
                                                            num_trials=num_trials, progress_bar=True)
    mean_tau_betting = StoppingTimesBetting.mean()
    NNBetting = np.arange(1, N_betting+1)
    deltaT = time() - t0 
    print(f'Sequential Betting test took {deltaT:.2f} seconds')
    ####=========================================================
    # Do the LC Test 
    print(f'\n====Starting LC test====\n')
    t0 = time()
    trials = num_trials # num_trials
    max_samples = N_LC # N_LC
    dataset = 'gmd'
    # dataset = 'gvd'
    # dataset = 'sg'
    theta0 = [0.5]
    maxTrials = 50
    alpha_size = 2
    ndims = d
    num_perturbations=num_pert_mean # num_perturb_mean
    Delta = epsilon_mean # epsilon_mean for gmd 
    trials_from = 0
    fixed_data_seed = False
    saveData=False
    data_gen_seed=False
    stop_when_reject=False

    PowerLC, StoppingTimesLC, NNLC = runLCexperiment(trials, max_samples, alpha, dataset, theta0, maxTrials, 
        alpha_size, saveData, data_gen_seed, stop_when_reject, ndims, 
        Delta, trials_from=0, fixed_data_seed=False, nTrees=40, max_rot_dim=0,
        ctw=False, local_rot=False, progress_bar=progress_bar, num_perturbations=num_perturbations)
    deltaT = time() - t0 
    ## division by 2 to count paired observations
    mean_tau_LC = StoppingTimesLC.mean()/2 
    print(f'Sequential LC test took {deltaT:.2f} seconds')
    ####=========================================================
    # Do the Sequential kernel-MMD test of Manole-Ramdas 
    t0 = time()
    print(f'\n====Starting MR test====\n')
    StoppingTimesMR, PowerMR = runMRTest(Source,  N=N_MR, num_trials=num_trials,
                                            kernel=kernel, B=1.0, alpha=alpha,
                                            parallel=False, progress_bar=progress_bar,
                                            return_type=1)
    NNMR = np.arange(1, N_MR+1)
    deltaT = time() - t0 
    mean_tau_MR = StoppingTimesMR.mean()
    print(f'Sequential MR test took {deltaT:.2f} seconds')

    ####=========================================================
    # Do the Sequential linear-time kernel-MMD test of Balsubramani-Ramdas 
    t0 = time()
    print(f'\n====Starting BR test====\n')
    StoppingTimesBR, PowerBR = runLilTest(Source, N=N_BR, num_trials=num_trials,
                                            kernel=kernel, alpha=alpha,
                                            parallel=False,  progress_bar=progress_bar,
                                            return_type=1)
    NNBR = np.arange(1, N_BR+1)
    mean_tau_BR = StoppingTimesBR.mean()
    deltaT = time() - t0 
    print(f'Sequential BR test took {deltaT:.2f} seconds')
    ## Prepare the data for plotting 
    Data = {}
    Data['batch'] = (PowerBatch, NNBatch) 
    Data['betting']=(PowerBetting, NNBetting, mean_tau_betting, StoppingTimesBetting) 
    Data['LC']=(PowerLC, NNLC, mean_tau_LC, StoppingTimesLC) 
    Data['MR']=(PowerMR, NNMR, mean_tau_MR, StoppingTimesMR) 
    Data['BR']=(PowerBR, NNBR, mean_tau_BR, StoppingTimesBR) 

    return Data 

def plot_results(Data, title, xlabel, ylabel, savefig=False, figname=None):
    palette = sns.color_palette(palette='husl', n_colors=10)
    plt.figure() 
    i=0
    for method in Data:
        if method=='batch':
            power, NN = Data[method] 
            plt.plot(NN, power, label=method, color=palette[i])
        else:
            power, NN, mean_tau, _ = Data[method]
            plt.plot(NN, power, label=method, color=palette[i])
            plt.axvline(x=mean_tau, linestyle='--', color=palette[i])
        i+=1 
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15)
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
    parser.add_argument('--N_batch', '-Nb', default=200, type=int)
    parser.add_argument('--num_perms', '-np', default=200, type=int)
    parser.add_argument('--num_trials', '-nt', default=500, type=int)
    parser.add_argument('--save_fig', '-sf', action='store_true')
    parser.add_argument('--save_data', '-sd', action='store_true')
    parser.add_argument('--alpha', '-a', default=0.05, type=float)

    args = parser.parse_args()

    d= args.d #10
    epsilon_mean = args.eps #0.5
    N_batch = args.N_batch #200
    N_betting = 3*N_batch 

    num_perms = args.num_perms #200
    num_trials = args.num_trials #500
    alpha= args.alpha #0.05

    num_steps_batch=  20
    ### parameters of the Gaussian distribution 
    epsilon_var = 0 
    num_pert_mean=1 
    num_pert_var = 0

    # Flags to save the data 
    savefig=args.save_fig
    savedata=args.save_data

    # run the experiment
    DataToPlot = main(N_batch=N_batch, N_betting=N_betting, d=d,
                    num_perms=num_perms,num_trials=num_trials,
                    alpha=alpha, epsilon_mean=epsilon_mean,
                    epsilon_var=epsilon_var, num_pert_mean=num_pert_mean, 
                    num_pert_var=num_pert_var, num_steps_batch=num_steps_batch,
                    progress_bar=True)
    title='Power vs Sample Size'
    xlabel='Sample-Size (n)'
    ylabel='Power'
    # get the path of the file to store data 
    parent_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    data_dir = parent_dir + '/data'
    figname = f'{data_dir}/Experiment1.png'
    plot_results(DataToPlot, title, xlabel, ylabel, savefig=savefig, figname=figname)

    filename = f'{data_dir}/Experiment1data.pkl'
    with open(filename, 'wb') as handle: 
        pickle.dump(DataToPlot, handle)