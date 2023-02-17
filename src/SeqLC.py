"""
Implementation of the Kdswitch based sequentail test of Lheritier-Cazals (2019)
This is a lightly modified version of the implementation written by Alix Lheritier
in the repository: https://github.com/alherit/kd-switch.git
"""
from __future__ import print_function

import time

import numpy as np
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import matplotlib.pyplot as plt
import math


import LCcommons  as cm
import LClogweightprob as logpr
from scipy.stats import special_ortho_group


import random
import sys

from tqdm import tqdm






#########################################################################################################
## DATA
#########################################################################################################
def tr_te_indices(n, tr_proportion, seed=9282 ):
    """Get two logical vectors for indexing train/test points.

    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion*n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)

def subsample_ind(n, k, seed=28):
    """
    Return a list of indices to choose k out of n without replacement
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    ind = np.random.choice(n, k, replace=False)
    np.random.set_state(rand_state)
    return ind


class TSTData(object):
    """Class representing data for two-sample test"""

    """
    properties:
    X, Y: numpy array 
    """

    def __init__(self, X, Y, label=None):
        """
        :param X: n x d numpy array for dataset X
        :param Y: n x d numpy array for dataset Y
        """
        self.X = X
        self.Y = Y
        # short description to be used as a plot label
        self.label = label

        nx, dx = X.shape
        ny, dy = Y.shape

        #if nx != ny:
        #    raise ValueError('Data sizes must be the same.')
        if dx != dy:
            raise ValueError('Dimension sizes of the two datasets must be the same.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0) 
        mean_y = np.mean(self.Y, 0)
        std_y = np.std(self.Y, 0) 
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n'%(np.array_str(mean_x, precision=prec ) )
        desc += 'E[y] = %s \n'%(np.array_str(mean_y, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
        desc += 'Std[y] = %s \n' %(np.array_str(std_y, precision=prec))
        return desc

    def dimension(self):
        """Return the dimension of the data."""
        dx = self.X.shape[1]
        return dx

    def dim(self):
        """Same as dimension()"""
        return self.dimension()

    def stack_xy(self):
        """Stack the two datasets together"""
        return np.vstack((self.X, self.Y))

    def xy(self):
        """Return (X, Y) as a tuple"""
        return (self.X, self.Y)

    def mean_std(self):
        """Compute the average standard deviation """

        #Gaussian width = mean of stds of all dimensions
        X, Y = self.xy()
        stdx = np.mean(np.std(X, 0))
        stdy = np.mean(np.std(Y, 0))
        # mstd = old_div((stdx + stdy),2.0)
        mstd = (stdx+stdy)/2.0
        return mstd
        #xy = self.stack_xy()
        #return np.mean(np.std(xy, 0)**2.0)**0.5
    
    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. Assume n is the same 
        for both X, Y. 
        
        Return (TSTData for tr, TSTData for te)"""
        X = self.X
        Y = self.Y
        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Require nx = ny')
        Itr, Ite = tr_te_indices(nx, tr_proportion, seed)
        label = '' if self.label is None else self.label
        tr_data = TSTData(X[Itr, :], Y[Itr, :], 'tr_' + label)
        te_data = TSTData(X[Ite, :], Y[Ite, :], 'te_' + label)
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new TSTData """
        if n > self.X.shape[0] or n > self.Y.shape[0]:
            raise ValueError('n should not be larger than sizes of X, Y.')
        ind_x = subsample_ind( self.X.shape[0], n, seed )
        ind_y = subsample_ind( self.Y.shape[0], n, seed )
        return TSTData(self.X[ind_x, :], self.Y[ind_y, :], self.label)

    ### end TSTData class  

class SampleSource(with_metaclass(ABCMeta, object)):
    """A data source where it is possible to resample. Subclasses may prefix 
    class names with SS"""

    @abstractmethod
    def sample(self, n, seed):
        """Return a TSTData. Returned result should be deterministic given 
        the input (n, seed)."""
        raise NotImplementedError()

    @abstractmethod
    def dim(self):
        """Return the dimension of the problem"""
        raise NotImplementedError()

    def visualize(self, n=400):
        """Visualize the data, assuming 2d. If not possible graphically,
        subclasses should print to the console instead."""
        data = self.sample(n, seed=1)
        x, y = data.xy()
        d = x.shape[1]

        if d==2:
            plt.plot(x[:, 0], x[:, 1], '.r', label='X')
            plt.plot(y[:, 0], y[:, 1], '.b', label='Y')
            plt.legend(loc='best')
        else:
            # not 2d. Print stats to the console.
            print(data)


class mySSSameGauss(SampleSource):
    """Two same standard Gaussians for P, Q. The null hypothesis 
    H0: P=Q is true."""
    def __init__(self, d):
        """
        d: dimension of the data 
        """
        self.d = d

    def dim(self):
        return self.d

    def sample(self, n, seed=None):
        rstate = np.random.get_state() #save current state
        if seed is not None:
            np.random.seed(seed) #use given seed to generate
        elif self.rstate is not None: #continue generation with previously saved state
            np.random.set_state(self.rstate)


        d = self.d
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d) 

        self.rstate = np.random.get_state() #keep current state to continue generation later
        np.random.set_state(rstate) #restore previous state

        return TSTData(X, Y, label='sg_d%d'%self.d)

class mySSGaussMeanDiff(SampleSource):
    """Toy dataset one in Chwialkovski et al., 2015. 
    P = N(0, I), Q = N( (my,0,0, 000), I). Only the first dimension of the means 
    differ."""
    def __init__(self, d, my=1.0, num_perturbations=1):
        """
        d: dimension of the data 
        """
        self.d = d
        self.my = my
        # dimension must be larger than the number of perturbations
        assert d>=num_perturbations
        self.num_perturbations=num_perturbations

    def dim(self):
        return self.d

    def sample(self, n, seed=None):
        rstate = np.random.get_state() #save current state
        if seed is not None:
            np.random.seed(seed) #use given seed to generate
        elif self.rstate is not None: #continue generation with previously saved state
            np.random.set_state(self.rstate)


        d = self.d
        num_perturbations = self.num_perturbations
        my = self.my
        # mean_y = np.hstack((self.my, np.zeros(d-1) ))
        mean_y = np.zeros((d,))
        mean_y[:num_perturbations] = my
        # X is zero mean observations 
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d) + mean_y
        
        self.rstate = np.random.get_state() #keep current state to continue generation later
        np.random.set_state(rstate) #restore previous state

        return TSTData(X, Y, label='gmd_d%d'%self.d)

class mySSGaussVarDiff(SampleSource):
    """Toy dataset two in Chwialkovski et al., 2015. 
    P = N(0, I), Q = N(0, diag((2, 1, 1, ...))). Only the variances of the first 
    dimension differ."""

    def __init__(self, d=5, dvar=2.0, num_perturbations=1):
        """
        d: dimension of the data 
        dvar: variance of the first dimension. 2 by default.
        """
        assert d>=num_perturbations
        self.d = d
        self.dvar = dvar
        self.num_perturbations=num_perturbations

    def dim(self):
        return self.d

    def sample(self, n, seed=None):
        rstate = np.random.get_state() #save current state
        if seed is not None:
            np.random.seed(seed) #use given seed to generate
        elif self.rstate is not None: #continue generation with previously saved state
            np.random.set_state(self.rstate)

        d = self.d
        dvar = self.dvar
        num_perturbations=self.num_perturbations
        std_y = np.diag(np.hstack(
            (np.ones(num_perturbations)*np.sqrt(dvar), np.ones(d-num_perturbations) )
        ))
        X = np.random.randn(n, d)
        Y = np.random.randn(n, d).dot(std_y)
        np.random.set_state(rstate)

        self.rstate = np.random.get_state() #keep current state to continue generation later
        np.random.set_state(rstate) #restore previous state


        return TSTData(X, Y, label='gvd')
        
#########################################################################################################
#########################################################################################################



## Bayesian forest
## each tree uses a random rotation
class SeqForest(object):
    def __init__(self, J, dim, alpha_label, theta0=None, keepItems=False, max_rot_dim = None, local_alpha= True, ctw=False, global_rot=True):

        self.max_rot_dim = max_rot_dim
                
        self.local_alpha = local_alpha
        self.J = J
        self.trees = []
        self.ctw =ctw

        for i in range(J):
            self.trees.append( SeqTree(dim=dim, alpha_label=alpha_label, theta0=theta0,
                                       keepItems=keepItems, max_rot_dim = max_rot_dim, local_alpha=local_alpha, ctw=ctw)
            )
    
            # initialize with uniform weights
            self.weights = np.repeat( logpr.LogWeightProb(1./J), J)
            


    def predict(self, point, label, update=True):
        '''
        Give prob of label given point, using CT*
        '''
        
        assert update # mandatory for kdSwitch
                
        prob_trees_mixture = logpr.LogWeightProb(0.)
        #MIX USING POSTERIOR
        for i in range(self.J):
            prob_tree = self.trees[i].predictUpdateKDSwitch(point,label) 
            prob_trees_mixture += prob_tree * self.weights[i]

        acc = logpr.LogWeightProb(0.)
        for i in range(self.J):
            self.weights[i] = self.trees[i].root.CTprob * logpr.LogWeightProb(1./self.J)
            acc += self.weights[i]

        for i in range(self.J):
            self.weights[i] /= acc

        return prob_trees_mixture


class SeqTree(object):

    def __init__(self, dim, alpha_label, theta0 = None, local_alpha=True,
                 keepItems = False, max_rot_dim = None, ctw=False):

            self.ctw = ctw
        
            self.max_rot_dim = max_rot_dim
            self.local_alpha = local_alpha

            if max_rot_dim > 0:
                #random rotation matrix
                # print("generating random rotation matrix...")
                if dim > max_rot_dim:
                    # print("using "+ str(max_rot_dim) + "random axes")
                    rot_axes = np.random.choice(range(dim),max_rot_dim,replace=False)
                    self.rot_mask = [True if i in rot_axes else False for i in range(dim)]  
                    rot_dim = max_rot_dim
                else:
                    self.rot_mask = None
                    rot_dim = dim
    
                self.R =  special_ortho_group.rvs(rot_dim)
            else:
                self.R = None
                # print("No rotation.")


            self.theta0 = theta0

            self.dim = dim
            self.alpha_label =  alpha_label
            
            #set this value to avoid learning global proportion
            if self.theta0 is not None:
                self.P0Dist = [logpr.LogWeightProb(p) for p in theta0]
            else:
                self.P0Dist = None

            self.n = 0

            #keep items in each node for post-hoc analysis: high memory consumption
            self.keepItems = keepItems

            self.root = SeqNode(depth=0,tree=self)
            



    def alpha(self,n):
        if self.ctw: #never switches
            return logpr.LogWeightProb(0.) 
        else:
            return logpr.LogWeightProb(1.)/logpr.LogWeightProb(n)

    def predictUpdateKDSwitch(self, point, label):
        
        #rotate
        if self.R is not None:
            if self.rot_mask is not None:
                rotated = np.dot(self.R, point[self.rot_mask])
                point[self.rot_mask] = rotated
            else:
                point = np.dot(self.R, point)
                
            
        self.n += 1
        return self.root.predictUpdateKDSwitch(point,label,updateStructure=True)


class SeqNode(object):
    
    def __init__(self, depth=0, tree=None ):

            self.depth = depth
            self.tree = tree

            self.projDir = np.random.randint(0,self.tree.dim) 

            self.items = [] #samples observed in this node. if not self.tree.keep_items, they are moved to children node when created
            
            self.Children = None #references to children of this node

            self.pivot = None #splitting point

            self.counts = [0 for x in range(self.tree.alpha_label)]

            # CTProb is the prob on whole seq giving by doing cts on this node (=mixing with children)
            self.CTprob = logpr.LogWeightProb(1.)

            ## for CTS
            self.wa = logpr.LogWeightProb(.5)
            self.wb = logpr.LogWeightProb(.5)

 
       
    ## main difference with original version: tree can be indefinitely refined: thus at current leaves, we must assume implicit children predicting as leaf (kt)    
    ## in the original version nodes can be adde, but leaf at depth D will remain leaf (this is why k and s don't matter at this node)
    def predictUpdateKDSwitch(self,point,label, updateStructure=True):

        splitOccurredHere = False
        
        #STEP 1: UPDATE STRUCTURE if leaf (full-fledged kd tree algorithm)
        if updateStructure and self.Children is None:

            splitOccurredHere = True
            
            self.pivot = self.computeProj(point)
            
            self.Children = [
                    self.__class__(depth=self.depth+1, tree=self.tree),
                    self.__class__(depth=self.depth+1, tree=self.tree)
                    ]
            
            for p in self.items: ### make all the update steps, notice that we are not yet adding current point
                i = self._selectBranch(p.point)
                self.Children[i].items.append(cm.LabeledPoint(point=p.point,label=p.label))
                self.Children[i].counts[p.label] += 1


            i = self._selectBranch(point)
            self.Children[i].items.append(cm.LabeledPoint(point=point,label=label)) #add current point to children's collection but not to label count, it will be done after prediction


            #initialize children using already existing symbols
            for i in [0,1]:
                self.Children[i].CTprob = logpr.LogWeightProb(log_wp=-cm.KT(self.Children[i].counts,alpha = self.tree.alpha_label))                  
                self.Children[i].wa *= self.Children[i].CTprob
                self.Children[i].wb *= self.Children[i].CTprob


            if not self.tree.keepItems:
                self.items = []  #all items have been sent down to children, so now should be empty
            
            
        #STEP 2 and 3: PREDICT AND UPDATE
        #save CTS_prob before update (lt_n = <n , cts prob up to previous symbol)        
        prob_CTS_lt_n = self.CTprob
        
        if self.depth ==0 and self.tree.P0Dist is not None: #known distribution
            prob_KT_next = self.tree.P0Dist[label]
        else:
            prob_KT_next = logpr.LogWeightProb(cm.seqKT(self.counts,label,alpha = self.tree.alpha_label)) #labels in self.items are not used, just counts are used 


        #now we can UPDATE the label count       
        self.counts[label] += 1


        if self.Children is None: 
            #PREDICT
            self.CTprob*=prob_KT_next #just KT

            #UPDATE
            self.wa*= prob_KT_next
            self.wb*= prob_KT_next
        else:
            #PREDICT (and recursive UPDATE)
            i = self._selectBranch(point)
            pr = self.Children[i].predictUpdateKDSwitch(point,label, updateStructure=not splitOccurredHere) 
            
            self.CTprob = self.wa * prob_KT_next + self.wb * pr

            #UPDATE
            if self.tree.local_alpha:
                alpha_n_plus_1 = self.tree.alpha(sum(self.counts)+1)
            else:
                alpha_n_plus_1 = self.tree.alpha(self.tree.n+1)

            self.wa = alpha_n_plus_1 * self.CTprob + (logpr.LogWeightProb(1.)- logpr.LogWeightProb(2.)*alpha_n_plus_1)* self.wa * prob_KT_next;
            self.wb = alpha_n_plus_1 * self.CTprob + (logpr.LogWeightProb(1.)- logpr.LogWeightProb(2.)*alpha_n_plus_1)* self.wb * pr;

       
        prob_CTS_up_to_n = self.CTprob
        
        return prob_CTS_up_to_n / prob_CTS_lt_n 
        
        
    def computeProj(self,point):
        if type(point)==dict: #sparse rep
            return point.get(self.projDir, 0.)
        else:
            return point[self.projDir]
    

    def _selectBranch(self, point):

        D = self.computeProj(point)

        if D <= self.pivot:
            return 0
        else:
            return 1
        
        #########################################################################################################
## Main Experiment
#########################################################################################################



class TST:

    
    def __init__(self, max_samples, alpha, dataset, theta0, maxTrials,
                  trial_n=None, saveData=False,  alpha_size=2, data_gen_seed=None, stop_when_reject=False, 
                  ndims=5, Delta=0.2, num_perturbations=1):

        self.tstData = None

        self.stop_when_reject = stop_when_reject 

        if data_gen_seed == None:
            data_gen_seed = trial_n
            
        self.start_time = None

        self.alpha_size = alpha_size

        # self.probs_file = open("./probs_seq_"+str(trial_n)+".dat","w")


        self.gadgetSampler = random.Random()
        self.gadgetSampler.seed(trial_n)

        self.dataSampler = random.Random()
        self.dataSampler.seed(data_gen_seed)


        #set global seed
        random.seed(trial_n)
        np.random.seed(trial_n)

        #self.min_samples = min_samples
        self.max_samples = max_samples

        self.alpha = alpha

        self.maxTrials = maxTrials

        self.theta0 = theta0
        self.cumTheta0 = np.cumsum(self.theta0)


        self.synthGen = None
        self.seqIndex = None
        
        self.unlimitedData = False
        
        if dataset is None  or dataset=='gmd':
            if ndims is  None:
                self.dim = 5
            else:
                self.dim = ndims
                
            meanshift = Delta
            
            self.synthGen = mySSGaussMeanDiff(d=self.dim, my=meanshift, 
                                                num_perturbations=num_perturbations)
            
            self.sample_as_gretton(data_gen_seed)

            self.xi = 0 #next sample to consume
            self.yi = 0 #next sample to consume

        elif dataset=='gvd':
            if ndims is  None:
                self.dim = 5
            else:
                self.dim = ndims
                
            dvar = Delta
            self.synthGen = mySSGaussVarDiff(d=self.dim, dvar=dvar, 
                                                num_perturbations=num_perturbations)

            self.sample_as_gretton(data_gen_seed)

            self.xi = 0 #next sample to consume
            self.yi = 0 #next sample to consume
            
        elif dataset=="sg":
            if ndims is  None:
                self.dim = 50
            else:
                self.dim = ndims
            
            self.synthGen = mySSSameGauss(d=self.dim)
            
            self.sample_as_gretton(data_gen_seed)
            
            self.xi = 0 #next sample to consume
            self.yi = 0 #next sample to consume

        else:
            exit("Wrong dataset")



        #sets of row indexes already sampled
        #this is used to ensure sampling without replacement
        self.sampled = {}
        for i in range(self.alpha_size):
            self.sampled[i] = set()

        self.processed = list()
 

        self.model = None # will be set later
        
        #here will be stored the p-value resulting from the two-sample test
        self.pvalue = logpr.LogWeightProb(1.)

    def sample_as_gretton(self,data_gen_seed):
        n=round(self.max_samples/2)
        self.tstData = self.synthGen.sample(n=n,seed=data_gen_seed)
        (self.X,self.Y) = self.tstData.xy()
        #then generate an extra 20% since the datasets are consumed randomly
        (X,Y) = self.synthGen.sample(n=round(n*.2)).xy()
        self.X = np.concatenate([self.X,X])
        self.Y = np.concatenate([self.Y,Y])
        
        

        
    def set_start_time(self):
        self.start_time = time.process_time()

    def __del__(self):
        # self.probs_file.close()
        pass 
        
    def setModel(self,model):
        self.model = model

    def sampleCat(self):
        u = self.gadgetSampler.uniform(0,1)
        return np.searchsorted(self.cumTheta0, u) 

    def getSample(self,pop=None):
        if pop is None:
            pop =  self.sampleCat()
            

        if self.synthGen is not None:
            if self.unlimitedData:
                point = self.synthGen.get_sample(pop)
            
            else:
                if pop==0:
                    if self.xi < len(self.X):
                        point = self.X[self.xi]
                        self.xi+=1
                    else:
                        return None
                else:
                    if self.yi < len(self.Y):
                        point = self.Y[self.yi]
                        self.yi+=1
                    else:
                        return None

        else:
            #let's try maxTrials times
            for i in range(self.maxTrials):
                row = self.dataSampler.randrange(0,self.datasets[pop].shape[0]-1)
                if not row in self.sampled[pop]:
                    break
                else:
                    row = -1

            if row==-1:
                print('Tried %i times. All rejected. ' % self.maxTrials)
                return None

            point = self.datasets[pop][row,:]

            self.sampled[pop].add(row)


        return cm.LabeledPoint(point,pop)

    def predictTheta0(self,label):
        if self.theta0 is not None:
            return logpr.LogWeightProb(self.theta0[label])
        else:
            return logpr.LogWeightProb(0.)



    def tst(self):

#        #cumulate in log form
        cumCProb = logpr.LogWeightProb(1.)
        cumTheta0Prob = logpr.LogWeightProb(1.)
        self.pvalue = logpr.LogWeightProb(1.)

        # convert to log form
        alpha = logpr.LogWeightProb(self.alpha)


        i = 1

        reject = False
 

        while (self.max_samples is None or i <= self.max_samples):
            
            if self.seqIndex is not None:
                if self.seqIndex<self.seqFeatures.shape[0]:
                    lp = cm.LabeledPoint(self.seqFeatures[self.seqIndex],self.seqLabels[self.seqIndex])
                    self.seqIndex+=1
                else:
                    lp = None
            else:
                lp = self.getSample()

            if lp is None:
                # print('No more samples.')
                # if not reject:
                    # print("No difference detected so far.")
                break


            condProb = self.model.predict(lp.point,lp.label,update=True)
            theta0Prob = self.predictTheta0(lp.label)

            cumCProb *= condProb
            cumTheta0Prob *= theta0Prob


            self.pvalue =  cumTheta0Prob/cumCProb #min(1.0,math.pow(2,log_theta0Prob-log_CTWProb))
            n = len(self.processed)+1
            # if n%10 ==0 :
                # print ('n=',n,'p-value', self.pvalue, 'alpha', self.alpha)

            nll = -cumCProb.getLogWeightProb()/n
            # self.probs_file.write(str(time.process_time()-self.start_time)+" "+str(condProb)+" "+str(nll)+ "\n")
            

            self.processed.append(lp)

            i += 1
            
            if not reject and self.pvalue <= alpha :
                reject = True
                n_reject = n
                p_value_reject = self.pvalue
                
                if self.stop_when_reject and reject:
                    break
        
            
        if not reject:
            n_ = n
            p_value_ = self.pvalue
            self.stopping_time = self.max_samples + 1 # did not reject.  
        else:
            n_ = n_reject
            p_value_ = p_value_reject
            self.stopping_time=n_reject+1 # actual number of paired observations 
        
        # print ('n=',n_,'p-value', p_value_, 'alpha', self.alpha)
    
        # print ('n=',n,'norm_log_loss', nll)


def get_rejection_rate(StoppingTimes, max_samples):
    num_trials = len(StoppingTimes) 
    R = np.zeros((max_samples, )) 
    for s in StoppingTimes:
        l = int(max(min(s-1, max_samples), 1))
        R[l:] = R[l:] + 1
    return R/num_trials


def runLCexperiment(trials, max_samples, alpha, dataset, theta0, maxTrials, 
        alpha_size, saveData, data_gen_seed, stop_when_reject, ndims, 
        Delta, trials_from=0, fixed_data_seed=False, nTrees=50, max_rot_dim=500,
        ctw=False, local_rot=True, progress_bar=True, num_perturbations=1):
    #add last probability
    if theta0 is not None:
        theta0.append(1.-sum(theta0)) 
        # print("theta0:",options.theta0)
        alpha_size = len(theta0) 
    else:
        alpha_size = alpha_size
    
    StoppingTimes = np.zeros((trials,))
    range_ = tqdm(range(trials)) if progress_bar else range(trials)
    for i in range_:
        trial_n = i+trials_from
        
        if fixed_data_seed:
            data_gen_seed = 0
        else:
            data_gen_seed = trial_n

        tst = TST(max_samples,
                alpha,
                dataset, 
                theta0,
                maxTrials, 
                trial_n=trial_n,
                saveData=False,
                alpha_size=alpha_size, 
                data_gen_seed = data_gen_seed,
                stop_when_reject=stop_when_reject,
                ndims=ndims,
                Delta=Delta, 
                num_perturbations=num_perturbations
                )

        # USE the KDswitch model of Lheritier-Cazals~(2019)
        tst.setModel ( SeqForest(J=nTrees, dim=tst.dim, alpha_label=alpha_size, 
                               theta0=theta0, keepItems = False, max_rot_dim = max_rot_dim, ctw=ctw, global_rot= not local_rot))
        # run the sequential test in trial i 
        tst.tst()

        StoppingTimes[i] = tst.stopping_time

    
    Power = get_rejection_rate(StoppingTimes, max_samples)
    NN = np.arange(2, max_samples+2)/2.0 # number of paired observations
    return Power, StoppingTimes, NN

 