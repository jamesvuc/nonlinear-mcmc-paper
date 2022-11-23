import random
import sys, datetime as dt
import numpy as np
import tqdm

import haiku as hk
import tree as dm_tree

import jax.numpy as jnp
import jax

from jax_bayes.mcmc import rms_langevin_fns, langevin_fns

sys.path.append('../../src/')
from nonlin_mcmc_fns import (
    tree_split_keys,
    tree_bg_select,
    tree_ar_propose,
    tree_ar_select,
    tree_jump_update
)
from custom_logging import Logger

from cifar10_dataloader import get_cifar10_dataset

np.random.seed(0)
random.seed(0)

def make_continuous_iter(dl):
    it = iter(dl)
    while True:
        try:
            yield from it
        except StopIteration:
            it = iter(dl)

bn_config = dict(
  create_scale=True,
  create_offset=True,
  decay_rate=0.999,
  eps=1e-5
)

initial_conv_config = dict()
initial_conv_config["output_channels"] = 64
initial_conv_config["kernel_shape"] = 7
initial_conv_config["stride"] = 2
initial_conv_config["with_bias"] = False
initial_conv_config["padding"] = "SAME"

#define the net-function
def net_fn(batch, is_training=True):
  net = hk.nets.ResNet18(10, #num classes
                        resnet_v2=True,
                        bn_config=bn_config,
                        initial_conv_config=initial_conv_config)
  x = batch['image']
  return net(x, is_training)

def make_mcmc_step_fn(sampler, logprob):
    def sampler_step_fn(i, sampler_state, keys, hk_state, batch):
        params = sampler.get_params(sampler_state)

        logp = lambda p,k: logprob(p, k, batch)
        (fx, hk_state), dx = jax.vmap(jax.value_and_grad(logp, has_aux=True))(params, hk_state)

        sampler_prop_state, new_keys = sampler.propose(
            i, dx, sampler_state, keys)
        # placeholder values for langevin/rms-langevin
        fx_prop, dx_prop = fx, dx

        accept_idxs, new_keys = sampler.accept(
            i, fx, fx_prop, dx, sampler_state, dx_prop,
            sampler_prop_state, new_keys)

        sampler_state, new_keys = sampler.update(
            i, accept_idxs, dx, sampler_state, dx_prop,
            sampler_prop_state, new_keys)

        return sampler_state, new_keys, hk_state
    return sampler_step_fn

def run(seed=0):
    logger = Logger(log_dir='dev_runs/')
    # logger = Logger(log_dir='runs/')

    #hyperparameters
    batch_size = 256
    base_lr = 1e-3
    aux_lr = lambda i: base_lr * (0.1**jnp.floor_divide(i,2_000))
    tgt_lr = 5e-5
    lr = aux_lr # for logging
    reg = 1e-4
    num_samples = 10
    noise_scale = 1e-4
    jump_prob = lambda i: 0.05
    aux_inv_temp = 0.9
    init_stddev = 1e-3
    no_tempering = False

    #instantiate the model --- same as regular case
    net = hk.transform_with_state(net_fn)

    #build the sampler instead of optimizer
    tgt_sampler_fns = langevin_fns
    aux_sampler_fns = rms_langevin_fns

    interaction_type = 'bg'
    # interaction_type = 'ar'

    logger.write(
        {
            'batch_size':batch_size,
            'base_lr':base_lr,
            'aux_lr':f'{base_lr} * (0.1**jnp.floor_divide(i,2_000))',
            'tgt_lr':tgt_lr,
            'reg':reg,
            'num_samples':num_samples,
            'noise_scale':noise_scale,
            'jump prob':'0.05',
            'aux_inv_temp':aux_inv_temp,
            'init_stdev':init_stddev,
            'tgt_sampler':'langevin_fns',
            'aux_sampler':'rms_lanveginV2_fns',
            'interaction_type':interaction_type,
            'tempered': not no_tempering
        }
    )

    key = jax.random.PRNGKey(seed)

    tgt_sampler = tgt_sampler_fns(
        key,
        num_samples=num_samples,
        step_size=tgt_lr,
        init_stddev=init_stddev,
        noise_scale=1.0 if no_tempering else noise_scale)

    aux_sampler = aux_sampler_fns(
        key,
        num_samples=num_samples,
        step_size=aux_lr,
        init_stddev=init_stddev,
        noise_scale=noise_scale # always temper the aux sampler
    )

    # loss is the same as the regular case.
    def loss(params, hk_state, batch):
        logits, hk_state = net.apply(params, hk_state, None, batch, is_training=True)
        labels = jax.nn.one_hot(batch['label'], 10)

        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p))
                            for ((mod_name, _), p) in dm_tree.flatten_with_path(params)
                            if 'batchnorm' not in mod_name)
        softmax_crossent = - jnp.mean(labels * jax.nn.log_softmax(logits))

        return softmax_crossent + reg * l2_loss, hk_state

    #the log-probability is the negative of the loss
    def logprob(p,k,b):
        lossval, state = loss(p, k, b)
        return -lossval, state

    def aux_logprob(*args):
        logp, state = logprob(*args)
        return aux_inv_temp * logp, state

    @jax.jit
    def accuracy(params, hk_state, batch):
        pred_fn = lambda p,s: net.apply(p, s, None, batch, is_training=False)
        pred_fn = jax.vmap(pred_fn)
        logits_samples, _ = pred_fn(params, hk_state)
        logits = jnp.mean(logits_samples, axis=0)
        return jnp.mean(jnp.argmax(logits, axis=-1) == batch['label'])

    #build the mcmc step. This is like the opimization step, but for sampling
    aux_sampler_step = make_mcmc_step_fn(aux_sampler, aux_logprob)
    tgt_sampler_step = make_mcmc_step_fn(tgt_sampler, logprob)

    @jax.jit
    def mcmc_step(i, state, aux_state, keys, hk_state, aux_hk_state, batch):
        aux_state, keys, aux_hk_state = aux_sampler_step(
            i, aux_state, keys, aux_hk_state, batch)
        state, keys, hk_state = tgt_sampler_step(
            i, state, keys, hk_state, batch)

        logp = lambda p,hs: logprob(p, hs, batch)[0]
        aux_logp = lambda p, hs: aux_logprob(p, hs, batch)[0]
        log_G = lambda p, hs: logp(p, hs) - aux_logp(p, hs)
        log_alpha = lambda p_x, p_y, hs_x, hs_y: \
                        log_G(p_y, hs_y) - log_G(p_x, hs_x)

        params = tgt_sampler.get_params(state)
        fx = jax.vmap(logp)(params, hk_state)

        if interaction_type == 'bg':
            y_aux = aux_sampler.get_params(aux_state)
            logG_aux = jax.vmap(log_G)(y_aux, aux_hk_state)

            keys, global_keys = tree_split_keys(keys, num_keys=3)
            global_sel_key, global_jump_key = global_keys
            sel_state, sel_aux_hk_state = \
                tree_bg_select(i, logG_aux, state, aux_state, global_sel_key,
                            addtl_data=aux_hk_state, state_has_aux=True)
        elif interaction_type == 'ar':
            keys, (global_prop_key,
                   global_sel_key,
                   global_jump_key) = tree_split_keys(keys, num_keys=4)

            aux_prop_state, sel_aux_hk_state = tree_ar_propose(
                i, num_samples, state, aux_state, global_prop_key,
                addtl_data=aux_hk_state, state_has_aux=True)

            zprop = tgt_sampler.get_params(aux_prop_state)
            x = tgt_sampler.get_params(state)
            log_alpha_aux = jax.vmap(log_alpha)(
                x, zprop, hk_state, aux_hk_state)

            sel_state = tree_ar_select(
                i, log_alpha_aux, state, aux_prop_state, global_sel_key)

        U = jax.random.uniform(global_jump_key, fx.shape)
        jump_idxs = U < jump_prob(i)

        state, hk_state = tree_jump_update(jump_idxs, state, sel_state,
                                           aux=hk_state, other_aux=sel_aux_hk_state)
        return jnp.mean(fx), state, aux_state, keys, hk_state, aux_hk_state

    # create batch iterators
    train_batches, test_batches = get_cifar10_dataset(batch_size)
    init_batches = iter(train_batches)
    init_batch = next(init_batches)

    # make the eval step
    def eval_step(params, hk_state):
        tot_acc = 0
        for i, batch in enumerate(test_batches):
            tot_acc += accuracy(params, hk_state, batch)
        tot_acc = tot_acc / (i+1)
        return tot_acc

    ##### Initialization
    # make batches of initialization keys
    all_keys = jax.random.split(jax.random.PRNGKey(1), 2 * num_samples + 2)
    keys = all_keys[:num_samples]
    aux_keys = all_keys[num_samples:2*num_samples]
    tgt_init_key = all_keys[-2]
    aux_init_key = all_keys[-1]

    # do the intialization of the samples of parameters
    init_param, _ = net.init(tgt_init_key, init_batch, is_training=True)
    aux_init_param, _ = net.init(tgt_init_key, init_batch, is_training=True)
    # the hk_states for the samples have to be initialized separately
    _, hk_state = jax.vmap(lambda k:net.init(k, init_batch, is_training=True))(keys)
    _, aux_hk_state = jax.vmap(lambda k:net.init(k, init_batch, is_training=True))(aux_keys)

    # initialize the samplers
    aux_sampler_state, _ = aux_sampler.init(aux_init_param)
    sampler_state, sampler_keys = tgt_sampler.init(init_param)

    # iterate the the Markov chain
    tot_steps = 0
    for epoch in tqdm.trange(50):
        for i, batch in enumerate(train_batches):
            (train_logprob,
            sampler_state,
            aux_sampler_state,
            sampler_keys,
            hk_state,
            aux_hk_state) = mcmc_step(
                tot_steps,
                sampler_state,
                aux_sampler_state,
                sampler_keys,
                hk_state,
                aux_hk_state,
                batch)
            tot_steps += 1

        if epoch % 1 == 0:
            params = tgt_sampler.get_params(sampler_state)
            aux_params = aux_sampler.get_params(aux_sampler_state)
            test_acc = eval_step(params, hk_state)
            aux_test_acc = eval_step(aux_params, aux_hk_state)
            print(
                f"epoch = {epoch} | total steps = {tot_steps}"
                f" | train logprob = {train_logprob:.3f}"
                f" | test acc = {test_acc:.3f}",
                f" | aux test acc = {aux_test_acc:.3f}",
                f" | lr={lr if isinstance(lr, float) else lr(tot_steps):.6f}"
            )

            logger.write(
                {
                    'epoch':f'{epoch}',
                    'tot_steps':f'{tot_steps}',
                    'test_acc':f'{test_acc}',
                    'aux_test_acc':f'{aux_test_acc}',
                    'aux_lr':f'{lr if isinstance(lr, float) else lr(tot_steps)}',
                    'timestamp':dt.datetime.now().strftime("%Y-%M-%d_%H:%m:%S")
                }
            )
            logger.dump(params, 'tgt_params')
            logger.dump(hk_state, 'hk_state')
            logger.dump(aux_params, 'aux_params')
            logger.dump(aux_hk_state, 'aux_hk_state')

    print('RUN DONE')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-tempering', action='store_true')
    ARGS = parser.parse_args()
    run(ARGS.seed)

if __name__ == '__main__':
    main()