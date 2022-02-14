# Nonlinear MCMC
This repo contains the code for the experiments in the paper ***Long-Time Convergence and Propoagation of Chaos for Nonlinear MCMC*** [arXiv:2202.05621](https://arxiv.org/abs/2202.05621). 

## Overview
In the paper, and in this repo, there are two main experiments: 
1. sampling from various multimodal 2-dimensional distributions; and 
2. Bayesian neural networks (BNNs) for MNIST digit classification. 

These experiments are written in python, relying heavily on the [JAX library](https://github.com/google/jax) from Google. Additionally, the BNN experiment uses the [haiku library](https://github.com/deepmind/dm-haiku) from DeepMind to implement the neural networks. The basic MCMC algorithms are implemented using the [jax-bayes library](https://github.com/jamesvuc/jax-bayes), so the actual [nonlinear mcmc code](./experiments/nonlin_mcmc_fns.py) is quite light. See [`experiments/`](./experiments) for the code associated to each experiment.

## Setup
You can use the script [`setup_env.sh`](./setup_env.sh) to set up a virtualenv (but not create one) with the dependencies required. Installing JAX can be a bit finicky depending on your compute environment, so YMMV. This was tested with python3.8 and cuda 11.2.

## Running the Experiments
Very simple: just run the script in the desired experiment directory with python (commandline args, etc.). For example, in `experiments/2d` run `python3 2d_exp.py`.

## Citing
If you use this work, please cite the paper
```
@article{vuckovic2022long,
  title={Long-Time convergence and propoagation of chaos for nonlinear mcmc},
  author={Vuckovic, James},
  journal={arXiv preprint arXiv:2202.05621},
  year={2022}
}
```

## License
[MIT License](./LICENSE)
