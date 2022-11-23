import sys
import tqdm

import haiku as hk
import tree as dm_tree

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from cifar10_dataloader import get_cifar10_dataset, make_continuous_iter

sys.path.append('../../src')
from custom_logging import Logger

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

def main():

    logger = Logger(log_dir='deterministic_runs/')

    # hyperparameters
    base_lr = 1e-3
    lr = lambda i: base_lr * (0.1**jnp.floor_divide(i,3_000))
    reg = 1e-4
    batch_size = 256

    logger.write(
        {
            'batch_size':batch_size,
            'lr':"lambda i: base_lr * (0.1**jnp.floor_divide(i,3_000))",
            'reg':reg,
            'epochs':50
        }
    )

    # haiku.transform turns the function into a 'haiku.Transformed' object
    # with init(...) and apply(...) methods.
    net = hk.transform_with_state(net_fn)

    #this is the closest to the rms_langevin algorithm
    opt_init, opt_update, opt_get_params = optimizers.rmsprop(lr)

    def loss(params, state, batch):
        logits, state = net.apply(params, state, None, batch, is_training=True)
        labels = jax.nn.one_hot(batch['label'], 10)

        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p))
                            for ((mod_name, _), p) in dm_tree.flatten_with_path(params)
                            if 'batchnorm' not in mod_name)
        softmax_crossent = - jnp.mean(labels * jax.nn.log_softmax(logits))

        return softmax_crossent + reg * l2_loss, state

    @jax.jit
    def accuracy(params, state, batch):
        preds, _ = net.apply(params, state, None, batch, is_training=False)
        return jnp.mean(jnp.argmax(preds, axis=-1) == batch['label'])

    @jax.jit
    def train_step(i, opt_state, state, batch):
        params = opt_get_params(opt_state)
        _loss = lambda p: loss(p, state, batch)
        (fx, state), dx = jax.value_and_grad(_loss, has_aux=True)(params)
        opt_state = opt_update(i, dx, opt_state)
        return fx, opt_state, state


    # load the data into memory and create batch iterators
    train_batches, test_batches = get_cifar10_dataset(batch_size)
    init_batches = iter(train_batches)

    @jax.jit
    def eval_step(params, hk_state):
        tot_acc = 0
        for i, batch in enumerate(test_batches):
            tot_acc += accuracy(params, hk_state, batch)
        tot_acc = tot_acc / (i+1)
        return tot_acc

    # instantiate the model parameters --- requires a 'test batch' for tracing
    params, state = net.init(jax.random.PRNGKey(42), next(init_batches), is_training=True)
    print(f"num params = {sum(jnp.size(p) for p in jax.tree_leaves(params)):,}")

    # intialize the optimzier state
    opt_state = opt_init(params)

    tot_steps = 0
    for epoch in tqdm.trange(51):
        for batch in train_batches:
            tot_steps += 1
            # run an optimization step
            train_loss, opt_state, state = train_step(tot_steps, opt_state, state, batch)

        if epoch % 5 == 0:
            params = opt_get_params(opt_state)
            test_acc = eval_step(params, state)
            print(f"epoch = {epoch} | tot steps = {tot_steps:,}"
                f" | lr = {lr(tot_steps):.6f}"
                f" | train loss = {train_loss:.4f}"
                f" | test acc = {test_acc:.3f}")

            logger.write(
                {
                    'train_loss':f'{train_loss}',
                    'epoch':f'{epoch}',
                    'test_acc':f'{test_acc}',
                    'lr':f'{lr(tot_steps)}'
                }
            )
            logger.dump(params, 'params')
            logger.dump(state, 'hk_state')

if __name__ == '__main__':
    main()