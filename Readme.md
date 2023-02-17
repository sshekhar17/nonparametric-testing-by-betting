## Code for reproducing the results in the paper "[Nonparametric two-sample testing by betting](arxiv.org/abs/2112.09162)"

**Summary:** we present a general framework for constructing powerful sequential two-sample tests by using the principle of testing by betting. In this repository, we include two instantiations of our general testing method: 
1. A sequential kernel-MMD test that works with multivariate or structured observations (`kernelMMD.py`) 
2. A sequential Kolmogorov-Smirnov test that works with real-valued data (`KStest.py`)

In addition to these, we also include implementations of the batch versions of kernel-MMD and KS tests, along with some other sequential tests such as those proposed by [Manole-Ramdas](https://arxiv.org/abs/2103.09267), [Balsubramani-Ramdas](https://arxiv.org/abs/1506.03486), [Darling-Robbins](https://www.jstor.org/stable/58954), and [Lheritier-Cazals](https://hal.inria.fr/hal-01135608v2/document). The Lheritier-Cazals test included in this repository is a very minor modification of [the python code written by Alix Lheritier](https://github.com/alherit/kd-switch), to allow interfacing with our data sources.

## Generating the results
The three experiments reported in Section 6 of the manuscript can be repeated by running  the following three files: 
* `python3 Experiment1.py --save_fig --save_data`
* `python3 Experiment2.py --save_fig --save_data`
* `python3 Experiment3.py --save_fig --save_data`

The two flags (`--save_fig` and `--save_data`) are used for saving the figures (in .png format) and the experiment data (in .pkl format). 