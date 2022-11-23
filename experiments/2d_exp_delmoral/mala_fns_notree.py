import math, collections

import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import make_schedule

centered_uniform = \
    lambda *args, **kwargs: jax.random.uniform(*args, **kwargs) - 0.5
init_distributions = dict(normal=jax.random.normal,
                          uniform=centered_uniform)

def match_dims(src, target, start_dim=1):
    """
    returns an array with the data from 'src' and same number of dims
    as 'target' by padding with empty dimensions starting at 'start_dim'
    """
    new_dims = tuple(range(start_dim, len(target.shape)))
    return jnp.expand_dims(src, new_dims)

SamplerFns = collections.namedtuple("SamplerFns", 
        ['init', 'propose', 'log_proposal', 'update', 'accept', 'get_params'])

def mala_fns(
    num_samples=10,
    step_size=1e-3,
    init_stddev=0.1,
    noise_scale=1.0,
    init_dist='normal'
):
    """Constructs sampler functions for the Metropolis Adjusted Langevin Algorithm.
    See e.g. http://probability.ca/jeff/ftpdir/lang.pdf
    Args:
        base_key: jax.random.PRNGKey
        num_samples: number of samples to initialize; either > 0 or -1.
            If num_samples == -1, assumes that the initial samples are
            already constructed.
        step_size: float or callable w/ signature step_size(i)
        init_stdev: nonnegative float standard deviation for initialization
            of initial samples (ignored if num_samples == -1)
        init_dist: str in ['normal', 'centered_uniform'] to sample perturbations
            for the initial distribution
    Returns:
        sampler function tuple (init, propose, update, get_params)
    """
    step_size = make_schedule(step_size)
    noise_scale = make_schedule(noise_scale)

    if isinstance(init_dist, str):
        init_dist = init_distributions[init_dist]

    def log_proposal(i, g, x, gprop, xprop): #grads come first
        #computes log q(xprop|x)
        return - 0.5 * jnp.sum(jnp.square((xprop - x - step_size(i) * g)) \
                / math.sqrt(2 *step_size(i) * noise_scale(i)))
    log_proposal = jax.vmap(log_proposal, in_axes=(None, 0, 0, 0, 0))

    def init(x0, key):
        init_key, next_key = jax.random.split(key)
        if num_samples == -1:
            return x0, next_key
        x = init_dist(init_key, (num_samples,) + x0.shape)
        return x0 + init_stddev * x, next_key

    def propose(i, g, x, key, **kwargs):
        key, next_key = jax.random.split(key)
        Z = jax.random.normal(key, x.shape)
        # return x + step_size(i) * g + math.sqrt(2 * step_size(i)) * Z, next_key
        return x + step_size(i) * g + math.sqrt(2 * step_size(i)) * noise_scale(i) * Z, next_key

    def update(i, accept_idxs, g, x, gprop, xprop, key):
        """ if the state had additional data, you would need to accept them too"""
        accept_idxs = match_dims(accept_idxs, x)
        mask = accept_idxs.astype(jnp.float32)

        xnext = x * (1.0 - mask) + xprop * mask
        return xnext, key

    def accept(i, logp, logp_prop, g, x, gprop, xprop, key):
        # compute logQ(xprop|x)
        logq_prop = log_proposal(i, g, x, gprop, xprop)

        # compute #logQ(x|xprop)
        logq_x = log_proposal(i, gprop, xprop, g, x)
        
        # compute log_alpha = log(P(xprop)/P(x) * Q(x|xprop)/Q(xprop|x))
        log_alpha = logp_prop + logq_x - logp - logq_prop
        
        # split a single key from the tree keys for the global acceptance step
        next_key, key = jax.random.split(key)
        
        # perform global acceptance step (sample accept/reject for each tree)
        # not each leaf of each tree
        U = jax.random.uniform(key, (logp.shape[0],))
        accept_idxs = jnp.log(U) < log_alpha
        
        return accept_idxs, next_key

    def get_params(x):
        return x

    return SamplerFns(init, propose, log_proposal, update, accept, get_params)