import os, json, sys, time
from matplotlib import pyplot as plt
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.scipy.stats.multivariate_normal as mvn

from mala_fns_notree import mala_fns
from nonlin_mcmc_fns_flat import (
    tree_bg_select_hist,
    tree_ar_select as tree_ar_select_hist,
    tree_ar_propose_hist,
    tree_jump_update
)

sys.path.append('../2d_exp')
from make_logprobs import ( 
    make_circular_mog, 
    make_ring_mixture,
    make_grid_mog,
    compute_mmd2
)

elementwise_value_and_grad = lambda f: jax.vmap(jax.value_and_grad(f))

def run(
    ax,
    lr=1e-3,
    jump_prob=0.1,
    interaction_type='bg',
    tgt_density='two_rings',
    n_samples=2_000,
    seed=None,
    title=None,
    other_ax=None,
    save_dir=None
):
    #====== Setup =======
    init_vals = jnp.array([0.0, 0.0])
    aux_init_vals = jnp.array([0.0, 0.0])

    if tgt_density == 'two_rings':
        dist = make_ring_mixture()
        aux_tgt_std = 4
    elif tgt_density == 'circular_mog':
        dist = make_circular_mog()
        aux_tgt_std = 4
    elif tgt_density == 'grid_mog':
        dist = make_grid_mog()
        aux_tgt_std = 20
    else:
        raise ValueError(f"Invalid tgt_density {tgt_density}!")
    logprob = dist.logprob

    aux_mean, aux_sigma = jnp.zeros(2), jnp.eye(2) * aux_tgt_std
    aux_logprob = jax.jit(lambda z: mvn.logpdf(z, mean=aux_mean, cov=aux_sigma))
    
    log_G = lambda z: logprob(z) - aux_logprob(z)
    log_alpha = lambda x, y: logprob(y) - logprob(x)\
                             + aux_logprob(x) - aux_logprob(y)

    g_tgt = elementwise_value_and_grad(logprob)
    g_aux = elementwise_value_and_grad(aux_logprob)

    seed = int(time.time() * 1000) if seed is None else seed
    init_key = jax.random.PRNGKey(seed)

    init_key, aux_init_key, gt_key = jax.random.split(init_key, 3)

    tgt_sampler_fn = mala_fns
    tgt_sampler = tgt_sampler_fn(
        num_samples=n_samples, step_size=lr,
        init_dist='uniform', init_stddev=15.0)
    sampler_state, sampler_keys = tgt_sampler.init(init_vals, init_key)

    aux_sampler_fn = mala_fns
    aux_sampler = \
        aux_sampler_fn(num_samples=n_samples, step_size=lr,
                    init_dist='uniform', init_stddev=15.0)
    aux_sampler_state, aux_sampler_keys = aux_sampler.init(aux_init_vals, aux_init_key)

    num_iters = 10_000
    # num_iters = 1000
    # num_iters = 100

    aux_hist = jnp.zeros(
        (aux_sampler_state.shape[0], num_iters + 1, aux_sampler_state.shape[1]))
    aux_hist = aux_hist.at[:, 0].set(aux_sampler_state)
    aux_sampler_state = aux_hist
    mask = jnp.ones(num_iters + 1)

    print(aux_sampler_state.dtype, aux_sampler_state.shape)


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

        return fx, state, keys

    # @jax.jit
    def _aux_step_append(i, hist, keys):
        state = hist[:, i]
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
        hist = hist.at[:, i+1].set(state)
        return hist, keys

    @jax.jit
    def _nonlin_step(i, state, aux_state, keys, mask):
        # aux_state, keys = _aux_step(i, aux_state, keys)
        aux_state, keys = _aux_step_append(i, aux_state, keys)
        fx, state, keys = _tgt_step(i, state, keys)

        x = tgt_sampler.get_params(state)

        if interaction_type == 'bg':
            y_aux = aux_sampler.get_params(aux_state)
            logG_aux = jax.vmap(jax.vmap(log_G))(y_aux)
            
            keys, global_sel_key, global_jump_key = \
                jax.random.split(keys, 3)
            sel_state = tree_bg_select_hist(
                i, logG_aux, state, aux_state, global_sel_key, mask)
            U = jax.random.uniform(global_jump_key, (logG_aux.shape[0],))

        elif interaction_type == 'ar':
            (keys, global_prop_key, 
                   global_sel_key, 
                   global_jump_key) = jax.random.split(keys, 4)

            aux_prop_state = tree_ar_propose_hist(
                i, n_samples, aux_state, global_prop_key, mask)
            z = aux_sampler.get_params(aux_prop_state)
            
            log_alpha_aux = jax.vmap(log_alpha)(x, z)
            sel_state = tree_ar_select_hist(
                i, log_alpha_aux, state, aux_prop_state, global_sel_key)
            U = jax.random.uniform(global_jump_key, (log_alpha_aux.shape[0],))
        
        else:
            raise ValueError(f"Invalid interaction type {interaction_type}")

        jump_idxs = U < jump_prob_fn(i)
        state = tree_jump_update(jump_idxs, state, sel_state)
        return fx, state, aux_state, keys

    gt_samples = dist.sample(gt_key, n_samples=10_000)
    mmd2_fn = lambda s:compute_mmd2(s, gt_samples, alphas=[1.0, 2.0])
    mmd2_fn = jax.jit(mmd2_fn)

    all_mmds = []
    for i in tqdm(range(num_iters)):
        mask = mask.at[i].set(0.0)
        fx, sampler_state, aux_sampler_state, sampler_keys = \
            _nonlin_step(i, sampler_state, aux_sampler_state, sampler_keys, mask)
        
        if i % 10 == 0:
            curr_samples = tgt_sampler.get_params(sampler_state)
            mmd = mmd2_fn(curr_samples)
            all_mmds.append([i, float(mmd)])

    samples = tgt_sampler.get_params(sampler_state)
    aux_samples = aux_sampler.get_params(aux_sampler_state)

    print(all_mmds[-1])

    #====== Plotting =======
    if ax is not None:
        H, xpts, ypts, _ = ax.hist2d(samples[:,0], samples[:, 1], bins=100, range=((-6, 6), (-6, 6)))
        if title is not None:
            ax.set_title(title)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xticks([])
        ax.set_yticks([])

    if other_ax is not None:
        other_ax.plot(*zip(*all_mmds))

    if save_dir:
        jnp.save(
            os.path.join(save_dir, 'tgt_samples.npy'), 
            samples,
            allow_pickle=False
        )
        jnp.save(
            os.path.join(save_dir, 'aux_samples.npy'), 
            aux_samples,
            allow_pickle=False
        )
        with open(os.path.join(save_dir, 'all_mmds.csv'), 'w') as f:
            for row in all_mmds:
                f.write(','.join(str(x) for x in row) + '\n')

        with open(os.path.join(save_dir, 'hparams.json'), 'w') as f:
            f.write(json.dumps(
                {
                    'lr':lr,
                    'jump_prob':jump_prob,
                    'tgt_density':tgt_density,
                    'n_samples':n_samples,
                    'seed':seed,
                    'aux_tgt_std':aux_tgt_std,
                    'init_stdev': 15.0,
                    'init_dist': 'uniform',
                    'num_iters':num_iters,
                    'num_gt_samples':10_000
                }
            ))

    return samples

def test():
    f, ax = plt.subplots()
    _, ax2 = plt.subplots()

    lr = 1e-3
    # run(ax, lr=lr, jump_prob=0.1, interaction_type='ar', 
    #             n_samples=2_000, seed=0)
    N = 2_000
    # N = 10
    run(ax, lr=lr, jump_prob=0.1, 
        interaction_type='bg', 
        # interaction_type='ar',
        # tgt_density='two_rings',
        tgt_density='grid_mog',
        # tgt_density='circular_mog',
        n_samples=N, seed=0, other_ax=ax2)

    plt.show()

def do_single():
    f, axes = plt.subplots(3)
    _, ax2 = plt.subplots()

    lr = 1e-3
    run(axes[0], lr=lr, jump_prob=0.0, interaction_type='ar', 
                n_samples=2_000, seed=1, other_ax=ax2)
    run(axes[1], lr=lr, jump_prob=0.1, interaction_type='ar', 
                n_samples=2_000, seed=1, other_ax=ax2)
    run(axes[2], lr=lr, jump_prob=0.1, interaction_type='bg', 
                n_samples=2_000, seed=1, other_ax=ax2)

    ax2.legend(['linear', 'ar', 'bg'])

    plt.show()

def collect_data():
    for tgt_density in ['two_rings', 'circular_mog', 'grid_mog']:
        print(tgt_density)
        for seed in range(1, 6):
            collect_data_helper(tgt_density, seed, do_plot=False)

def collect_data_helper(tgt_density, seed, do_plot=False):
    if do_plot:
        f, axes = plt.subplots(3)
        _, ax2 = plt.subplots()
    else:
        axes = [None] * 3
        ax2 = None
    base_save_dir = f'./final_data_delmoral/{tgt_density}/seed{seed}/'

    save_dir = os.path.join(base_save_dir, 'linear')
    os.makedirs(save_dir)
    run(axes[0], lr=1e-3, jump_prob=0.0, interaction_type='ar', 
                n_samples=2_000, seed=seed, tgt_density=tgt_density,
                other_ax=ax2, save_dir=save_dir)
    
    save_dir = os.path.join(base_save_dir, 'ar')
    os.makedirs(save_dir)
    run(axes[1], lr=1e-3, jump_prob=0.1, interaction_type='ar', 
                n_samples=2_000, seed=seed, tgt_density=tgt_density,
                other_ax=ax2, save_dir=save_dir)
    
    save_dir = os.path.join(base_save_dir, 'bg')
    os.makedirs(save_dir)
    run(axes[2], lr=1e-3, jump_prob=0.1, interaction_type='bg', 
                n_samples=2_000, seed=seed, tgt_density=tgt_density,
                other_ax=ax2, save_dir=save_dir)

    if do_plot:
        ax2.legend(['linear', 'ar', 'bg'])
        plt.show()


if __name__ == '__main__':
    # test()
    do_single()
    # collect_data()