from functools import partial

import jax
import jax.numpy as jnp
from jax._src.util import safe_zip, safe_map

map = safe_map
zip = safe_zip

def match_dims(src, target, start_dim=1):
    """
    returns an array with the data from 'src' and same number of dims
    as 'target' by padding with empty dimensions starting at 'start_dim'
    """
    new_dims = tuple(range(start_dim, len(target.shape)))
    return jnp.expand_dims(src, new_dims)

def tree_bg_select(i, logG_y, tgt_state, aux_state, global_key, addtl_data=None):
    sel_idxs = jax.random.categorical(global_key, logG_y, shape=logG_y.shape)
    y_aux = aux_state
    select_fn = lambda idxs, y: y[idxs]
    ysel = select_fn(sel_idxs, y_aux)

    if addtl_data is not None:
        raise NotImplementedError
        # addtl_sel = jax.tree_map(
        #     lambda y: y[sel_idxs],
        #     addtl_data)
    
    if addtl_data is None:
        return ysel
    else:
        return ysel, addtl_sel

def tree_bg_select_hist(i, logG_y, tgt_state, aux_state, global_key, mask=None,
                    addtl_data=None):
    logG_y = logG_y - 100_000.0 * mask
    sel_idxs = jax.random.categorical(global_key, logG_y, axis=-1)
    y_aux = aux_state

    select_fn = lambda idxs, y: y[idxs]
    ysel = jax.vmap(select_fn)(sel_idxs, y_aux)

    if addtl_data is not None:
        raise NotImplementedError
        # addtl_sel = jax.tree_map(
        #     lambda y: y[sel_idxs],
        #     addtl_data)

    if addtl_data is None:
        return ysel
    else:
        return ysel, addtl_sel

def tree_ar_select(i, log_alpha, state, aux_state, global_key):
    U = jax.random.uniform(global_key, shape=log_alpha.shape)
    accept_idxs = jnp.log(U) < log_alpha

    x = state
    y_aux = aux_state

    def select_fn(idxs, x, y):
        idxs = match_dims(idxs, x)
        mask = idxs.astype(jnp.float32)
        return x * (1.0 - mask) + y * mask

    ysel = jax.vmap(select_fn)(accept_idxs, x, y_aux)
    return ysel


def tree_ar_propose(i, N, aux_state, global_key, addtl_data=None):
    """ this is basically the same as the BG select with uniform probs """
    prop_idxs = jax.random.randint(global_key, shape=(N,), minval=0, maxval=N)
    y_aux = aux_state
    select_fn = lambda idxs, y: y[idxs]
    zprop = select_fn(prop_idxs, y_aux)

    if addtl_data is not None:
        raise NotImplementedError
        # addtl_prop = jax.tree_map(
        #     lambda y: y[prop_idxs],
        #     addtl_data)
    return zprop

def tree_ar_propose_hist(i, N, aux_state, global_key, mask=None,
                        addtl_data=None):
    mask = -100_000.0 * mask
    prop_idxs = jax.random.categorical(
        global_key, mask, axis=-1, shape=(N,))
    
    y_aux = aux_state
    
    select_fn = lambda idxs, y: y[idxs]
    zprop = jax.vmap(select_fn)(prop_idxs, y_aux)
    
    if addtl_data is not None:
        raise NotImplementedError
        # addtl_prop = jax.tree_map(
        #     lambda y: y[prop_idxs],
        #     addtl_data)
    return zprop

def tree_jump_update(jump_idxs, state, other_state, aux=None, other_aux=None):
    assert (aux is None) == (other_aux is None)
    x = state
    y = other_state
    
    def select_fn(idxs, _x, _y):
        idxs = match_dims(idxs, _x)
        mask = idxs.astype(_x.dtype) # will this trace?
        return _x * (1 - mask) + _y * mask

    ysel = select_fn(jump_idxs, x, y)

    if aux is not None:
        raise NotImplementedError
        # sel_aux = jax.tree_map(
        #     partial(select_fn, jump_idxs),
        #     aux, other_aux
        # )
        # return SamplerState(ysel_flat, treedef, subtrees), sel_aux
        
    return ysel