
import jax
import jax.numpy as jnp
from jax._src.util import partial, safe_zip, safe_map, unzip2

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

def tree_bg_select(i, logG_y, tgt_state, aux_state, global_key, 
                    addtl_data=None, state_has_aux=False):
    sel_idxs = jax.random.categorical(global_key, logG_y, 
                                      shape=logG_y.shape)

    _, tgt_treedef, tgt_subtrees = tgt_state
    y_aux_flat, aux_treedef, aux_subtrees = aux_state
    y_aux = map(jax.tree_util.tree_unflatten, aux_subtrees, y_aux_flat)

    if state_has_aux:
        # assumes that SamplerState[0] is the actual state value
        select_fn = lambda idxs, y: y[0][idxs]
    else:
        select_fn = lambda idxs, y: y[idxs]
    ysel = map(partial(select_fn, sel_idxs), y_aux)
    ysel_flat, ysel_subtree = unzip2(map(jax.tree_util.tree_flatten, ysel))

    if addtl_data is not None:
        addtl_sel = jax.tree_map(
            lambda y: y[sel_idxs],
            addtl_data
        )

    if addtl_data is None:
        return SamplerState(ysel_flat, tgt_treedef, tgt_subtrees)
    else:
        return SamplerState(ysel_flat, tgt_treedef, tgt_subtrees), addtl_sel

def tree_ar_propose(i, N, tgt_state, aux_state, global_key, 
                    addtl_data=None, state_has_aux=False):
    prop_idxs = jax.random.randint(
        global_key, shape=(N,), minval=0, maxval=N)

    _, tgt_treedef, tgt_subtrees = tgt_state
    y_aux_flat, aux_treedef, aux_subtrees = aux_state
    y_aux = map(jax.tree_util.tree_unflatten, aux_subtrees, y_aux_flat)

    if state_has_aux:
        # assumes that SamplerState[0] is the actual state value
        select_fn = lambda idxs, y: y[0][idxs]
    else:
        select_fn = lambda idxs, y: y[idxs]
    zprop = map(partial(select_fn, prop_idxs), y_aux)
    zprop_flat, zprop_subtree = unzip2(map(jax.tree_util.tree_flatten, zprop))

    if addtl_data is not None:
        addtl_prop = jax.tree_map(
            lambda y: y[prop_idxs],
            addtl_data
        )

    if addtl_data is None:
        return SamplerState(zprop_flat, tgt_treedef, tgt_subtrees)
    else:
        return SamplerState(zprop_flat, tgt_treedef, tgt_subtrees), addtl_prop


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

def tree_jump_update(jump_idxs, state, other_state, aux=None, other_aux=None):
    assert (aux is None) == (other_aux is None)
    x_flat, treedef, subtrees = state
    x = map(jax.tree_util.tree_unflatten, subtrees, x_flat)

    y_flat, other_treedef, other_subtrees = other_state
    y = map(jax.tree_util.tree_unflatten, other_subtrees, y_flat)

    def select_fn(idxs, _x, _y):
        idxs = match_dims(idxs, _x)
        # mask = idxs.astype(jnp.float32)
        mask = idxs.astype(_x.dtype) # will this trace?
        return _x * (1 - mask) + _y * mask

    ysel = map(partial(select_fn, jump_idxs), x, y) # this is correct
    ysel_flat, ysel_subtree = unzip2(map(jax.tree_util.tree_flatten, ysel))

    if aux is not None:
        sel_aux = jax.tree_map(
            partial(select_fn, jump_idxs),
            aux, other_aux
        ) # this works

        return SamplerState(ysel_flat, treedef, subtrees), sel_aux
    
    return SamplerState(ysel_flat, treedef, subtrees)