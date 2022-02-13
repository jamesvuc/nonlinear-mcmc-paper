from functools import partial
import sys, os, math
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt

import haiku as hk

import jax.numpy as jnp
import jax

import jax_bayes
from jax_bayes.mcmc import mala_fns

sys.path.append('..')
from nonlin_mcmc_fns import (
    tree_split_keys,
    tree_bg_select,
    tree_ar_select,
    tree_jump_update
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow_datasets as tfds

#load the data and create the model
def load_dataset(split, is_training, batch_size):
    ds = tfds.load('mnist:3.*.*', split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))

def net_fn(batch):
	""" Standard LeNet-300-100 MLP """
	x = batch["image"].astype(jnp.float32) / 255.

	# we initialize the model with zeros since we're going to construct intiial 
	# samples for the weights with additive Gaussian noise
	sig = 0.0
	mlp = hk.Sequential([
		hk.Flatten(),
		hk.Linear(300, w_init=hk.initializers.RandomNormal(stddev=sig),
					   b_init=hk.initializers.RandomNormal(stddev=sig)), 
		jax.nn.relu, 
		hk.Linear(100, w_init=hk.initializers.RandomNormal(stddev=sig),
					   b_init=hk.initializers.RandomNormal(stddev=sig)), 
		jax.nn.relu, 
		hk.Linear(10,  w_init=hk.initializers.RandomNormal(stddev=sig),
					   b_init=hk.initializers.RandomNormal(stddev=sig))
		])

	return mlp(x)

# assumes constant lr for now
def log_proposal(lr, x, dx, y, dy):
    return - 0.5 * jnp.sum(jnp.square((y - x - lr * dx)) \
                                    / math.sqrt(2 *lr))

def main():
    #hyperparameters
    lr = 5e-3
    # lr = 1e-3
    reg = 1e-4
    num_samples = 100 # number of samples to approximate the posterior -- 
    init_stddev = 5.0 # initial distribution for the samples will be N(0, 0.1)
    num_steps = 10_001
    train_batch_size = 1_000
    eval_batch_size = 10_000
    jump_prob = lambda i: 0.05
    aux_inv_temp = 0.9
    noise_scale = 0.1
    
    #instantiate the model --- same as regular case
    net = hk.transform(net_fn)

    #build the sampler instead of optimizer
    tgt_sampler_fns = mala_fns
    aux_sampler_fns = mala_fns
    # seed = 0
    seed = 1 # for the different trial runs
    # seed = 2
    # seed = 3
    # seed = 4
    # seed = 5
    key = jax.random.PRNGKey(seed)
    
    tgt_sampler = tgt_sampler_fns(
        key, 
        num_samples=num_samples, 
        step_size=lr, 
        init_stddev=init_stddev,
        noise_scale=noise_scale
    )

    aux_sampler = aux_sampler_fns(
        key,
        num_samples=num_samples,        
        step_size=lr,
        init_stddev=init_stddev,
        noise_scale=noise_scale
    )

    # loss is the same as the regular case! This is because in regular ML, we're minimizing
    # the negative log-posterior logP(params | data) = logP(data | params) + logP(params) + constant
    # i.e. finding the MAP estimate.
    def loss(params, batch):
        logits = net.apply(params, jax.random.PRNGKey(0), batch)
        labels = jax.nn.one_hot(batch['label'], 10)
        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) 
                    for p in jax.tree_leaves(params))
        softmax_crossent = - jnp.mean(labels * jax.nn.log_softmax(logits))

        return softmax_crossent + reg * l2_loss

    #the log-probability is the negative of the loss
    logprob = lambda p,b : - loss(p, b)
    aux_logprob = lambda *args: aux_inv_temp * logprob(*args)

    @jax.jit
    def accuracy(params, batch):
        #auto-vectorize over the samples of params! only in JAX...
        pred_fn = jax.vmap(net.apply, in_axes=(0, None, None))

        # this is crucial --- integrate (i.e. average) out the parameters
        preds = jnp.mean(pred_fn(params, None, batch), axis=0)

        return jnp.mean(jnp.argmax(preds, axis=-1) == batch['label'])

    #build the mcmc step. This is like the opimization step, but for sampling
    def aux_sampler_step(i, state, keys, batch):
        params = aux_sampler.get_params(state)

        logp = lambda params: aux_logprob(params, batch)
        fx, dx = jax.vmap(jax.value_and_grad(logp))(params)
        
        prop_state, keys = aux_sampler.propose(i, dx, state, keys)

        prop_params = aux_sampler.get_params(prop_state)
        fx_prop, dx_prop = jax.vmap(jax.value_and_grad(logp))(prop_params)

        # generate the acceptance indices from the Metropolis-Hastings
        # accept-reject step
        accept_idxs, keys = aux_sampler.accept(
            i, fx, fx_prop, dx, state, dx_prop, prop_state, keys
        )

        # perform a short burn-in accepting every step
        accept_idxs = jax.lax.cond(
            i < 100,
            lambda idxs: jnp.ones_like(idxs),
            lambda idxs: idxs,
            accept_idxs
        )
        # update the sampler state based on the acceptance acceptance indices
        state, keys = aux_sampler.update(
            i, accept_idxs, dx, state, dx_prop, prop_state, keys
        )

        return state, keys

    def tgt_sampler_step(i, state, keys, batch):
        params = tgt_sampler.get_params(state)

        logp = lambda params: logprob(params, batch)
        fx, dx = jax.vmap(jax.value_and_grad(logp))(params)
        
        prop_state, keys = tgt_sampler.propose(i, dx, state, keys)

        prop_params = tgt_sampler.get_params(prop_state)
        fx_prop, dx_prop = jax.vmap(jax.value_and_grad(logp))(prop_params)

        accept_idxs, keys = tgt_sampler.accept(
            i, fx, fx_prop, dx, state, dx_prop, prop_state, keys
        )

        # perform a short burn-in accepting every step
        accept_idxs = jax.lax.cond(
            i < 100,
            lambda idxs: jnp.ones_like(idxs),
            lambda idxs: idxs,
            # lambda idxs: jnp.ones_like(idxs),
            accept_idxs
        )
        state, keys = tgt_sampler.update(
            i, accept_idxs, dx, state, dx_prop, prop_state, keys
        )

        return state, keys

    @jax.jit
    def mcmc_step(i, state, aux_state, keys, batch):
        aux_state, keys = aux_sampler_step(i, aux_state, keys, batch)
        state, keys = tgt_sampler_step(i, state, keys, batch)

        logp = lambda params: logprob(params, batch)
        aux_logp = lambda params: aux_logprob(params, batch)
        log_G = lambda p: logp(p) - aux_logp(p)

        params = tgt_sampler.get_params(state)
        fx = jax.vmap(logp)(params)

        y_aux = aux_sampler.get_params(aux_state)
        logG_aux = jax.vmap(log_G)(y_aux)
        
        keys, global_keys = tree_split_keys(keys, num_keys=3)
        global_sel_key, global_jump_key = global_keys
        sel_state = tree_bg_select(i, logG_aux, aux_state, global_sel_key)
        
        U = jax.random.uniform(global_jump_key, fx.shape)
        jump_idxs = U < jump_prob(i)
        
        state = tree_jump_update(jump_idxs, state, sel_state)
        return jnp.mean(fx), state, aux_state, keys

    # load the data into memory and create batch iterators
    train_batches = load_dataset("train", is_training=True, batch_size=train_batch_size)
    val_batches = load_dataset("train", is_training=False, batch_size=eval_batch_size)
    test_batches = load_dataset("test", is_training=False, batch_size=eval_batch_size)

    #get a single sample of the params using the normal hk.init(...)
    params = net.init(jax.random.PRNGKey(42), next(train_batches))

    # get a SamplerState object with `num_samples` params along dimension 0
    # generated by adding Gaussian noise (see sampler_fns(..., init_dist='normal'))
    # sampler_state, sampler_keys = sampler_init(params)
    aux_sampler_state, sampler_keys = aux_sampler.init(params)
    sampler_state, sampler_keys = tgt_sampler.init(params)

    # iterate the the Markov chain
    for step in trange(10_001):
        train_logprob, sampler_state, aux_sampler_state, sampler_keys = \
            mcmc_step(step, sampler_state, aux_sampler_state, sampler_keys, next(train_batches))
        if step % 500 == 0:
            params = tgt_sampler.get_params(sampler_state)
            val_acc = accuracy(params, next(val_batches))
            test_acc = accuracy(params, next(test_batches))
            print(f"step = {step}"
                f" | val acc = {val_acc:.3f}"
                f" | test acc = {test_acc:.3f}")

    def do_analysis():
        test_data = next(test_batches)
        pred_fn = jax.vmap(net.apply, in_axes=(0, None, None))

        all_test_logits = pred_fn(params, None, test_data)
        probs = jnp.mean(jax.nn.softmax(all_test_logits, axis=-1), axis=0)
        correct_preds_mask = jnp.argmax(probs, axis=-1) == test_data['label']

        entropies = jax_bayes.utils.entropy(probs)
        correct_ent = entropies[correct_preds_mask]
        incorrect_ent = entropies[~correct_preds_mask]

        mean_correct_ent = jnp.mean(correct_ent)
        mean_incorrect_ent = jnp.mean(incorrect_ent)

        plt.hist(np.array(correct_ent), alpha=0.3, label='correct', density=True)
        plt.hist(np.array(incorrect_ent), alpha=0.3, label='incorrect', density=True)
        plt.axvline(x=mean_correct_ent, color='blue', label='mean correct')
        plt.axvline(x=mean_incorrect_ent, color='orange', label='mean incorrect')
        plt.legend()
        plt.xlabel("entropy")
        plt.ylabel("histogram density")
        plt.title("posterior predictive entropy of correct vs incorrect predictions")
        plt.show()

    do_analysis()

if __name__ == '__main__':
    main()