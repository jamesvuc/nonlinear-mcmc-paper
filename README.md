# Nonlinear MCMC
This repo contains the code for the experiments in the NeurIPS 2022 publication ***Nonlinear MCMC for Bayesian Machine Learning*** [arXiv:2202.05621](https://arxiv.org/abs/2202.05621). 

## Overview
In the paper, and in this repo, there are two main experiments: 
1. sampling from various multimodal 2-dimensional distributions; and 
2. Bayesian neural networks (BNNs) for CIFAR10 object classification. 

These experiments are written in python, relying heavily on the [JAX library](https://github.com/google/jax) from Google. Additionally, the BNN experiment uses the [`haiku` library](https://github.com/deepmind/dm-haiku) from DeepMind to implement the neural networks. The basic MCMC algorithms are implemented using the [`jax-bayes` library](https://github.com/jamesvuc/jax-bayes), so the actual [nonlinear mcmc code](./src/nonlin_mcmc_fns.py) is quite light. See [`experiments/`](./experiments) for the code associated to each experiment.

## Setup
You can use the script [`setup_env.sh`](./setup_env.sh) to set up a virtualenv (but not create one) with the dependencies required. Installing JAX can be a bit finicky depending on your compute environment, so YMMV. This was tested with python3.8 and cuda 11.3.

## Running the Experiments
Very simple: just run the script in the desired experiment directory with python (commandline args, etc.). For example, in `experiments/2d_exp` run `python3 2d_exp.py`.

Here is an overview of the main experiment files (excluding helpers):

File | Description
--|--
|`experiments/2d_exp/2d_exp.py`| Main 2d experiment file|
| `experiments/2d_exp_delmoral/2d_exp_flat_delmoral.py` | Experiment file to simulate the algorithm from [this paper](https://www.stats.ox.ac.uk/~doucet/andrieu_jasra_doucet_delmoral_nonlinearMCMC.pdf) which is the inspiration for our work | 
| `experiments/2d_exp_delmoral/2d_exp_flat.py` | Re-implementation of `experiments/2d_exp/2d_exp.py` without `jax-bayes` or pytrees for a sanity check |
| `experiments/cifar10/cifar10_nonlin_resnet_mcmc.py` |  Main experiment file for CIFAR10 MCMC experiments |
| `experiments/cifar10/cifar10_resnet_deterministic.py` | Sanity check file for the CIFAR10 experiment using traditional ML |

## Citing
If you use this work, please cite the paper
```
@article{vuckovic2022nonlinear
  title={Nonlinear {MCMC} for {B}ayesian Machine Learning},
  author={Vuckovic, James},
  journal={arXiv preprint arXiv:2202.05621},
  year={2022}
}
```

## License
[MIT License](./LICENSE)
