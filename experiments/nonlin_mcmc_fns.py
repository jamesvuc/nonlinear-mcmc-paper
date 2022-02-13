import jax
import jax.numpy as jnp
from jax._src.util import partial, safe_zip, safe_map, unzip2

import sys
from jax_bayes.mcmc import SamplerState, SamplerKeys

from functools import partial

map = safe_map
zip = safe_zip

def match_dims(src, target, start_dim=1):
    """
    returns an array with the data from 'src' and same number of dims
    as 'target' by padding with empty dimensions starting at 'start_dim'
    """
    new_dims = tuple(range(start_dim, len(target.shape)))
    return jnp.expand_dims(src, new_dims)

def tree_split_keys(tree_keys, num_keys=2):
    assert num_keys >= 2
    keys_flat, keys_meta = jax.tree_util.tree_flatten(tree_keys)
    all_new_keys = jax.random.split(keys_flat[0], num_keys)
    global_keys, next_key = all_new_keys[:-1], all_new_keys[-1]
    keys_flat[0] = next_key
    return SamplerKeys(keys_flat), global_keys

def tree_bg_select(i, logG_y, aux_state, global_key):
    sel_idxs = jax.random.categorical(global_key, logG_y, 
                                      shape=logG_y.shape)
    
    y_aux_flat, aux_treedef, aux_subtrees = aux_state
    y_aux = map(jax.tree_util.tree_unflatten, aux_subtrees, y_aux_flat)

    select_fn = lambda idxs, y: y[idxs]

    ysel = map(partial(select_fn, sel_idxs), y_aux)
    ysel_flat, ysel_subtree = unzip2(map(jax.tree_util.tree_flatten, ysel))
    
    return SamplerState(ysel_flat, aux_treedef, aux_subtrees)
    
def tree_ar_select(i, log_alpha, state, aux_state, global_key):
    U = jax.random.uniform(global_key, shape=log_alpha.shape)
    accept_idxs = jnp.log(U) < log_alpha

    x_flat, treedef, subtrees = state
    x = map(jax.tree_util.tree_unflatten, subtrees, x_flat)

    y_aux_flat, aux_treedef, aux_subtrees = aux_state
    y_aux = map(jax.tree_util.tree_unflatten, aux_subtrees, y_aux_flat)

    def select_fn(idxs, x, y):
        idxs = match_dims(idxs, x)
        mask = idxs.astype(jnp.float32)
        return x * (1.0 - mask) + y * mask

    ysel = map(partial(select_fn, accept_idxs), x, y_aux)
    ysel_flat, ysel_subtree = unzip2(map(jax.tree_util.tree_flatten, ysel))
    
    return SamplerState(ysel_flat, aux_treedef, aux_subtrees)


def tree_jump_update(jump_idxs, state, other_state):
    x_flat, treedef, subtrees = state
    x = map(jax.tree_util.tree_unflatten, subtrees, x_flat)

    y_flat, other_treedef, other_subtrees = other_state
    y = map(jax.tree_util.tree_unflatten, other_subtrees, y_flat)

    def select_fn(idxs, x, y):
        idxs = match_dims(idxs, x)
        mask = idxs.astype(jnp.float32)
        return x * (1.0 - mask) + y * mask

    ysel = map(partial(select_fn, jump_idxs), x, y)
    ysel_flat, ysel_subtree = unzip2(map(jax.tree_util.tree_flatten, ysel))

    return SamplerState(ysel_flat, other_treedef, other_subtrees)