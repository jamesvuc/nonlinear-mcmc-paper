import numpy as onp

import jax
import jax.numpy as jnp
import jax.scipy.stats.multivariate_normal as mvn

import itertools, math
import datetime as dt
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm

from jax_bayes.mcmc import mala_fns

import sys; sys.path.append('..')
from nonlin_mcmc_fns import (
    tree_split_keys,
    tree_bg_select,
    tree_ar_select,
    tree_jump_update
)

from make_logprobs import make_two_moons, make_circular_mog, make_ring_mixture

elementwise_value_and_grad = lambda f: jax.vmap(jax.value_and_grad(f))

def run(
    ax,
    lr=1e-3,
    jump_prob=0.1,
    interaction_type='bg',
    n_samples=2_000,
    seed=None,
    title=None,
    tgt_logprob='moons'
):
    #====== Setup =======
    key = jax.random.PRNGKey(1)
    init_vals = jnp.array([0.0, 0.0])
    aux_init_vals = jnp.array([0.0, 0.0])

    if tgt_logprob == 'moons':
        logprob = make_two_moons()
    elif tgt_logprob == 'mog':
        logprob = make_circular_mog()
    elif tgt_logprob == 'rings':
        logprob = make_ring_mixture()
    else:
        raise ValueError(f"Got invalid tgt_logprob {tgt_logprob}!")

    # aux_logprob = lambda z: 0.7 * logprob(z)
    aux_mean, aux_sigma = jnp.zeros(2), jnp.eye(2) * 4
    aux_logprob = jax.jit(lambda z: mvn.logpdf(z, mean=aux_mean, cov=aux_sigma))
    log_G = lambda z: logprob(z) - aux_logprob(z)
    log_alpha = lambda x, y: logprob(y) - logprob(x)\
                             + aux_logprob(x) - aux_logprob(y)

    allsamps = []

    #====== Runs =======
    g = elementwise_value_and_grad(logprob)
    g_tgt = g
    g_aux = elementwise_value_and_grad(aux_logprob)

    seed = int(time.time() * 1000) if seed is None else seed
    init_key = jax.random.PRNGKey(seed)

    tgt_sampler_fn = mala_fns
    tgt_sampler = \
        tgt_sampler_fn(init_key, num_samples=n_samples, step_size=lr,
                     init_dist='uniform', init_stddev=15.0)
    sampler_state, sampler_keys = tgt_sampler.init(init_vals)
    init_state = sampler_state


    aux_sampler_fn = mala_fns
    aux_sampler = \
        aux_sampler_fn(init_key, num_samples=n_samples, step_size=lr,
                    init_dist='uniform', init_stddev=15.0)
    aux_sampler_state, aux_sampler_keys = aux_sampler.init(aux_init_vals)
    aux_init_state = aux_sampler_state

    jump_prob_fn = lambda i: jump_prob

    def _tgt_step(i, state, keys):
        x = tgt_sampler.get_params(state)
        fx, dx = g_tgt(x)

        prop_state, keys = tgt_sampler.propose(i, dx, state, keys)

        xprop = tgt_sampler.get_params(prop_state)
        fx_prop, dx_prop = jax.vmap(jax.value_and_grad(logprob))(xprop)

        accept_idxs, keys = tgt_sampler.accept(
            i, fx, fx_prop, dx, state, dx_prop, prop_state, keys
        )
        state, keys = tgt_sampler.update(
            i, accept_idxs, dx, state, dx_prop, prop_state, keys
        )
        return state, keys

    @jax.jit
    def _aux_step(i, state, keys):
        x = aux_sampler.get_params(state)
        fx, dx = g_aux(x)

        prop_state, keys = aux_sampler.propose(i, dx, state, keys)

        xprop = aux_sampler.get_params(prop_state)
        fx_prop, dx_prop = jax.vmap(jax.value_and_grad(aux_logprob))(xprop)

        accept_idxs, keys = aux_sampler.accept(
            i, fx, fx_prop, dx, state, dx_prop, prop_state, keys
        )
        state, keys = aux_sampler.update(
            i, accept_idxs, dx, state, dx_prop, prop_state, keys
        )
        return state, keys

    @jax.jit
    def _nonlin_step(i, state, aux_state, keys):
        aux_state, keys = _aux_step(i, aux_state, keys)
        state, keys = _tgt_step(i, state, keys)

        x = tgt_sampler.get_params(state)

        if interaction_type == 'bg':
            y_aux = aux_sampler.get_params(aux_state)
            logG_aux = jax.vmap(log_G)(y_aux)
            
            keys, global_keys = tree_split_keys(keys, num_keys=3)
            global_sel_key, global_jump_key = global_keys
            sel_state = tree_bg_select(i, logG_aux, aux_state, global_sel_key)
            U = jax.random.uniform(global_jump_key, logG_aux.shape)

        elif interaction_type == 'ar':
            y_aux = aux_sampler.get_params(aux_state)
            log_alpha_aux = jax.vmap(log_alpha)(x, y_aux)

            keys, global_keys = tree_split_keys(keys, num_keys=3)
            global_sel_key, global_jump_key = global_keys
            sel_state = tree_ar_select(i, log_alpha_aux, state, aux_state, global_sel_key)
            U = jax.random.uniform(global_jump_key, log_alpha_aux.shape)
        else:
            raise ValueError(f"Invalid interaction type {interaction_type}")

        jump_idxs = U < jump_prob_fn(i)

        state = tree_jump_update(jump_idxs, state, sel_state)
        return state, aux_state, keys

    num_iters = 5_000
    for i in tqdm(range(num_iters)):
        sampler_state, aux_sampler_state, sampler_keys = \
            _nonlin_step(i, sampler_state, aux_sampler_state, sampler_keys)

    samples = tgt_sampler.get_params(sampler_state)
    aux_samples = aux_sampler.get_params(aux_sampler_state)

    #====== Plotting =======

    H, xpts, ypts, _ = ax.hist2d(samples[:,0], samples[:, 1], bins=100, range=((-4, 4), (-4, 4)))
    
    if title is not None:
        ax.set_title(title)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_target(ax, tgt_logprob):
    if tgt_logprob == 'moons':
        logprob = make_two_moons()
    elif tgt_logprob == 'mog':
        logprob = make_circular_mog()
    elif tgt_logprob == 'rings':
        logprob = make_ring_mixture()
    else:
        raise ValueError(f"Got invalid tgt_logprob {tgt_logprob}!")
        
    pts = onp.linspace(-3, 3, 1000)
    X, Y = onp.meshgrid(pts, pts)
    pos = onp.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z = onp.exp(jax.vmap(jax.vmap(logprob))(pos))
    Z = Z/Z.max()
    Z = jnp.expand_dims(Z, axis=-1)
    
    ax.imshow(Z, extent=(X.min(), X.max(), Y.max(), Y.min()),
       interpolation='nearest', cmap=cm.viridis)

    ax.set_xticks([])
    ax.set_yticks([])

def run_lr_sweep():
    seed = 0
    # seed = 1
    interaction_types = ['bg', 'ar']
    lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    exponent = [-5, -4, -3, -2, -1]
    f, axes = plt.subplots(3, len(lrs))

    for j, (lr, expn) in enumerate(zip(lrs, exponent)):
        ax = axes[0][j]
        run(ax, lr=lr, jump_prob=0.0, n_samples=2_000, seed=seed,
            title=f"lr=1e{expn}")

    for i, it in enumerate(interaction_types):
        for j, lr in enumerate(lrs):
            ax = axes[i+1][j]
            run(ax, lr=lr, jump_prob=0.1, interaction_type=it, 
                n_samples=2_000, seed=seed)

    axes[0,0].set_ylabel('Linear')
    axes[1,0].set_ylabel('BG Interaction')
    axes[2,0].set_ylabel('AR Interaction')

    plt.show()

def run_overview():
    seed = 0
    interaction_types = ['bg', 'ar']
    lr = 1e-3
    f, axes = plt.subplots(3, 4, sharex=True, sharey=True)
    tgt_logprobs = ['moons', 'mog', 'rings']
    for j, tgt_logprob in enumerate(tgt_logprobs):
        plot_target(axes[j, 0], tgt_logprob=tgt_logprob) 

    for j, tgt_logprob in enumerate(tgt_logprobs):
        ax = axes[j][1]
        run(ax, lr=lr, jump_prob=0.0, interaction_type='ar', 
            n_samples=2_000, seed=seed, tgt_logprob=tgt_logprob)

    for i, it in enumerate(interaction_types):
        for j, tgt_logprob in enumerate(tgt_logprobs):
            ax = axes[j][i+2]
            run(ax, lr=lr, jump_prob=0.1, interaction_type=it, 
                n_samples=2_000, seed=seed, tgt_logprob=tgt_logprob)

    axes[0,0].set_title('Target')
    axes[0,1].set_title('Linear')
    axes[0,2].set_title('BG Interaction')
    axes[0,3].set_title('AR Interaction')

    plt.show()


if __name__ == '__main__':
    run_overview()
    # run_lr_sweep()
    # test()
