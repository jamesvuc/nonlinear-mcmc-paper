import haiku as hk

import jax.numpy as jnp
from jax.experimental import optimizers
import jax

import jax_bayes

import sys, os, math
import numpy as np
from tqdm import trange

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow_datasets as tfds

from matplotlib import pyplot as plt

#load the data and create the model
def load_dataset(split, is_training, batch_size):
    ds = tfds.load('mnist:3.*.*', split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))

def net_fn(batch):
    """ 
  Standard LeNet-300-100 MLP 
    """
    x = batch["image"].astype(jnp.float32) / 255.
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(300), jax.nn.relu, 
        hk.Linear(100), jax.nn.relu, 
        hk.Linear(10)])

    return mlp(x)


def main():

    # hyperparameters
    lr = 1e-3
    reg = 1e-4

    # haiku.transform turns the function into a 'haiku.Transformed' object
    # with init(...) and apply(...) methods.
    net = hk.transform(net_fn)

    #this is the closest to the rms_langevin algorithm
    opt_init, opt_update, opt_get_params = optimizers.rmsprop(lr)

    def loss(params, batch):
        logits = net.apply(params, None, batch)
        labels = jax.nn.one_hot(batch['label'], 10)

        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) 
                            for p in jax.tree_leaves(params))
      
        softmax_crossent = - jnp.mean(labels * jax.nn.log_softmax(logits))

        return softmax_crossent + reg * l2_loss

    @jax.jit
    def accuracy(params, batch):
        preds = net.apply(params, None, batch)
        return jnp.mean(jnp.argmax(preds, axis=-1) == batch['label'])

    @jax.jit
    def train_step(i, opt_state, batch):
        params = opt_get_params(opt_state)
        dx = jax.grad(loss)(params, batch)
        opt_state = opt_update(i, dx, opt_state)
        return opt_state

    # load the data into memory and create batch iterators
    train_batches = load_dataset("train", is_training=True, batch_size=1_000)
    val_batches = load_dataset("train", is_training=False, batch_size=10_000)
    test_batches = load_dataset("test", is_training=False, batch_size=10_000)


    # instantiate the model parameters --- requires a 'test batch' for tracing
    params = net.init(jax.random.PRNGKey(42), next(train_batches))

    # intialize the optimzier state
    opt_state = opt_init(params)


    for step in trange(301):
        if step % 100 == 0:
            params = opt_get_params(opt_state)
            val_acc = accuracy(params, next(val_batches))
            test_acc = accuracy(params, next(test_batches))
            print(f"step = {step}"
                  f" | val acc = {val_acc:.3f}"
                  f" | test acc = {test_acc:.3f}")
      
        opt_state = train_step(step, opt_state, next(train_batches))

    def do_analysis():
        test_data = next(test_batches)
        pred_fn = net.apply

        logits = pred_fn(params, None, test_data)
        probs = jax.nn.softmax(logits, axis=-1)
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