import os, json, time, sys
from matplotlib import pyplot as plt
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.scipy.stats.multivariate_normal as mvn

from jax_bayes.mcmc import mala_fns

sys.path.append('../../src')
from nonlin_mcmc_fns import (
    tree_split_keys,
    tree_bg_select,
    tree_ar_select,
    tree_ar_propose,
    tree_jump_update
)

from make_logprobs import (
    make_circular_mog,
    make_ring_mixture,
    make_grid_mog,
    compute_mmd2
)

elementwise_value_and_grad = lambda f: jax.vmap(jax.value_and_grad(f))

def make_mcmc_step_fn(sampler, val_and_grad_fn):
    def mcmc_step_fn(i, state, keys):
        x = sampler.get_params(state)
        fx, dx = val_and_grad_fn(x)

        prop_state, keys = sampler.propose(i, dx, state, keys)

        xprop = sampler.get_params(prop_state)
        fx_prop, dx_prop = val_and_grad_fn(xprop)

        accept_idxs, keys = sampler.accept(
            i, fx, fx_prop, dx, state, dx_prop, prop_state, keys)
        state, keys = sampler.update(
            i, accept_idxs, dx, state, dx_prop, prop_state, keys)

        return fx, state, keys
    return mcmc_step_fn

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
        init_key, num_samples=n_samples, step_size=lr,
        init_dist='uniform', init_stddev=15.0)
    sampler_state, sampler_keys = tgt_sampler.init(init_vals)

    aux_sampler_fn = mala_fns
    aux_sampler = aux_sampler_fn(
        aux_init_key, num_samples=n_samples, step_size=lr,
        init_dist='uniform', init_stddev=15.0)
    aux_sampler_state, _ = aux_sampler.init(aux_init_vals)

    jump_prob_fn = lambda i: jump_prob

    _tgt_step = make_mcmc_step_fn(tgt_sampler, g_tgt)
    _aux_step = make_mcmc_step_fn(aux_sampler, g_aux)

    @jax.jit
    def _nonlin_step(i, state, aux_state, keys):
        if interaction_type is not None:
            _, aux_state, keys = _aux_step(i, aux_state, keys)
        fx, state, keys = _tgt_step(i, state, keys)

        x = tgt_sampler.get_params(state)

        if interaction_type == 'bg':
            y_aux = aux_sampler.get_params(aux_state)
            logG_aux = jax.vmap(log_G)(y_aux)

            keys, global_keys = tree_split_keys(keys, num_keys=3)
            global_sel_key, global_jump_key = global_keys
            sel_state = tree_bg_select(i, logG_aux, state, aux_state, global_sel_key,)
            U = jax.random.uniform(global_jump_key, logG_aux.shape)

        elif interaction_type == 'ar':
            keys, (global_prop_key,
                   global_sel_key,
                   global_jump_key) = tree_split_keys(keys, num_keys=4)

            aux_prop_state = tree_ar_propose(
                i, n_samples, state, aux_state, global_prop_key)

            z = aux_sampler.get_params(aux_prop_state)
            log_alpha_aux = jax.vmap(log_alpha)(x, z)

            sel_state = tree_ar_select(
                i, log_alpha_aux, state, aux_prop_state, global_sel_key)
            U = jax.random.uniform(global_jump_key, log_alpha_aux.shape)

        elif interaction_type is None:
            return fx, state, aux_state, keys

        else:
            raise ValueError(f"Invalid interaction type {interaction_type}")

        jump_idxs = U < jump_prob_fn(i)

        state = tree_jump_update(jump_idxs, state, sel_state)
        return fx, state, aux_state, keys

    gt_samples = dist.sample(gt_key, n_samples=10_000)
    mmd2_fn = lambda s:compute_mmd2(s, gt_samples, alphas=[1.0, 2.0])
    mmd2_fn = jax.jit(mmd2_fn)

    all_mmds = []
    num_iters = 10_000
    _t = time.time()
    for i in tqdm(range(num_iters)):
        fx, sampler_state, aux_sampler_state, sampler_keys = \
            _nonlin_step(i, sampler_state, aux_sampler_state, sampler_keys)

        if i % 10 == 0:
            _delta_t = time.time() - _t
            curr_samples = tgt_sampler.get_params(sampler_state)
            mmd = mmd2_fn(curr_samples)
            all_mmds.append([i, mmd, _delta_t])

    samples = tgt_sampler.get_params(sampler_state)
    aux_samples = aux_sampler.get_params(aux_sampler_state)

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
        other_ax.plot(*list(zip(*all_mmds))[:2])

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

    return samples, all_mmds

def test():
    f, ax = plt.subplots()
    _, ax2 = plt.subplots()

    lr = 1e-3
    tgt_density = 'circular_mog'
    # tgt_density = 'two_rings'
    # tgt_density='grid_mog'

    # interaction_type = None
    interaction_type = 'bg'
    # interaction_type = 'ar'
    run(ax, lr=lr, jump_prob=0.1,
        interaction_type=interaction_type,
        n_samples=2_000, seed=0,
        tgt_density=tgt_density,
        other_ax=ax2)

    plt.show()

def do_single():
    f, axes = plt.subplots(3)
    _, ax2 = plt.subplots()

    lr = 1e-3
    # tgt_density = 'two_rings'
    # tgt_density = 'circular_mog'
    tgt_density = 'grid_mog'
    run(axes[0], lr=lr, jump_prob=0.0, interaction_type=None,
                n_samples=2_000, seed=1, other_ax=ax2,
                tgt_density=tgt_density)
    run(axes[1], lr=lr, jump_prob=0.1, interaction_type='ar',
                n_samples=2_000, seed=1, other_ax=ax2,
                tgt_density=tgt_density)
    run(axes[2], lr=lr, jump_prob=0.1, interaction_type='bg',
                n_samples=2_000, seed=1, other_ax=ax2,
                tgt_density=tgt_density)

    ax2.legend(['linear', 'ar', 'bg'])

    plt.show()

def collect_data():
    for tgt_density in ['two_rings', 'circular_mog', 'grid_mog']:
        for seed in range(1, 6):
            collect_data_helper(tgt_density, seed, do_plot=False)

def collect_data_helper(tgt_density, seed, do_plot=False):
    if do_plot:
        f, axes = plt.subplots(3)
        _, ax2 = plt.subplots()
    else:
        axes = [None] * 3
        ax2 = None

    base_save_dir = f'./final_data/{tgt_density}/seed{seed}/'
    save_dir = os.path.join(base_save_dir, 'linear')
    os.makedirs(save_dir)
    n_samples = 2_000
    # n_samples = 10_000
    run(axes[0], lr=1e-3, jump_prob=0.0, interaction_type=None,
                n_samples=n_samples, seed=seed, tgt_density=tgt_density,
                other_ax=ax2, save_dir=save_dir)

    save_dir = os.path.join(base_save_dir, 'ar')
    os.makedirs(save_dir)
    run(axes[1], lr=1e-3, jump_prob=0.1, interaction_type='ar',
                n_samples=n_samples, seed=seed, tgt_density=tgt_density,
                other_ax=ax2, save_dir=save_dir)

    save_dir = os.path.join(base_save_dir, 'bg')
    os.makedirs(save_dir)
    run(axes[2], lr=1e-3, jump_prob=0.1, interaction_type='bg',
                n_samples=n_samples, seed=seed, tgt_density=tgt_density,
                other_ax=ax2, save_dir=save_dir)

    if do_plot:
        ax2.legend(['linear', 'ar', 'bg'])
        plt.show()

def collect_scaling_data():
    ax, ax2 = None, None

    # tgt_density = 'grid_mog'
    tgt_density = 'two_rings'
    # interaction_type = 'bg'
    interaction_type = 'ar'

    sweep_range = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
    mmd_data = dict()
    for n_samples in sweep_range:
        _, mmd_res = run(ax, lr=1e-3, jump_prob=0.1, interaction_type=interaction_type,
                  n_samples=n_samples, seed=1, other_ax=ax2, tgt_density=tgt_density)
        mmd_data[n_samples] = [[i, x] for i, x, _ in mmd_res]

    import numpy as np
    avg_data = dict()
    for N, data in mmd_data.items():
        xs, ys = map(list, zip(*data))
        plt.plot(xs, ys, label=f"N={N}", alpha=0.7)

        avg_data[N] = np.mean(ys[-250:])

    f, ax = plt.subplots()
    tmp = [avg_data[N] for N in sweep_range]
    print(tmp)
    ax.plot(sweep_range, tmp)
    ax.scatter(sweep_range, tmp)

    plt.show()


if __name__ == '__main__':
    # test()
    do_single()
    # collect_data()
    # collect_scaling_data()