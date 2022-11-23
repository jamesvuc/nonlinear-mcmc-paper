from collections import namedtuple

import jax
import jax.numpy as jnp

Distribution = namedtuple('Distribution', ['logprob', 'sample'])

""" adapted from https://github.com/VincentStimper/normalizing-flows/blob/master/normflow/distributions.py"""

_norm = lambda x: jnp.sqrt(jnp.sum(jnp.square(x)))

def make_two_moons():

    def logprob(z):
        """
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        logp = - 0.5 * ((_norm(z) - 2) / 0.2) ** 2 \
                + jnp.log(jnp.exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                          + jnp.exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        return logp

    return Distribution(logprob, None)



def make_circular_mog(n_modes=8):
    pi = 3.141592658
    scale = 2 / 3 * jnp.sin(pi / n_modes)

    component_means = [(2 * jnp.sin(2 * pi / n_modes * i),
                        2 * jnp.cos(2 * pi / n_modes * i))
                        for i in range(n_modes)]

    def logprob(z):
        components = []
        for i in range(n_modes):
            # c = ((z[0] - 2 * jnp.sin(2 * pi / n_modes * i)) ** 2
            #       + (z[1] - 2 * jnp.cos(2 * pi / n_modes * i)) ** 2)\
            #      / (2 * scale ** 2)
            c = ((z[0] - component_means[i][0]) ** 2
                  + (z[1] - component_means[i][1]) ** 2)\
                 / (2 * scale ** 2)
            components.append(c)

        log_p = - jnp.log(2 * pi * scale ** 2 * n_modes) \
                + jax.scipy.special.logsumexp(-jnp.array(components))
        return log_p

    def sample(key, n_samples=1_000):
        key1, key2 = jax.random.split(key, 2)

        Z = jax.random.normal(key1, shape=(n_samples, 2)) * scale

        logits = jnp.array([1.] * len(component_means))
        component_idxs = jax.random.categorical(
            key2, logits, shape=(n_samples,))
        tmp_means = jnp.array(component_means)

        Z = Z + tmp_means[component_idxs]
        return Z

    # return logprob

    return Distribution(logprob, sample)


def make_ring_mixture(n_rings=2):
    scale = 1 / 4 / n_rings

    component_means = [2 / n_rings * (i + 1) for i in range(n_rings)]

    def logprob(z):
        components = []
        for i in range(n_rings):
            # c = ((_norm(z) - 2 / n_rings * (i + 1)) ** 2) \
            #      / (2 * scale ** 2)
            c = ((_norm(z) - component_means[i]) ** 2) / (2 * scale ** 2)
            components.append(c)
        return jax.scipy.special.logsumexp(-jnp.array(components))

    # return logprob

    def sample(key, n_samples=1_000):
        key1, key2, key3 = jax.random.split(key, 3)
        pi = 3.141592653

        r = jax.random.normal(key1, (n_samples,)) * scale
        theta = jax.random.uniform(key2, (n_samples,)) * 2 * pi # 0..2pi

        logits = jnp.array([1.] * n_rings)
        component_idxs = jax.random.categorical(
            key3, logits, shape=(n_samples,))
        tmp_means = jnp.array(component_means)
        r = r + tmp_means[component_idxs]

        xcoords = r * jnp.cos(theta)
        ycoords = r * jnp.sin(theta)
        return jnp.stack([xcoords, ycoords], axis=1)

    return Distribution(logprob, sample)

def make_grid_mog():
    pi = 3.141592658
    centers = [(i,j) for i in range(-4,5,2) for j in range(-4,5,2)]
    n_modes = len(centers)
    scale = 0.4

    def logprob(z):
        components = []
        for a,b in centers:
            c = ((z[0] - a) ** 2 + (z[1] - b) ** 2) / (2 * (scale ** 2))
            components.append(c)

        log_p = - jnp.log(2 * pi * (scale ** 2) * n_modes) \
                + jax.scipy.special.logsumexp(-jnp.array(components))
        return log_p

    def sample(key, n_samples=1_000):
        key1, key2 = jax.random.split(key, 2)

        Z = jax.random.normal(key1, shape=(n_samples, 2)) * scale

        logits = jnp.array([1.] * len(centers))
        component_idxs = jax.random.categorical(
            key2, logits, shape=(n_samples,))
        tmp_means = jnp.array(centers)

        Z = Z + tmp_means[component_idxs]
        return Z

    return Distribution(logprob, sample)

def compute_mmd2(sample1, sample2, alphas=[1.0, 2.0]):
    """
    compute the squared mmd between samples1 and samples2

    sample1 is of shape (num_samples1, d) and
    sample2 is of shape (num_samples2, d)

    adapted from https://github.com/necludov/particle-EBMs/blob/8a4c01d3b87f5b27e87fce5aa0a0eaa439e4aabd/utils/utils.py#L75
    for more info see eq (3) of https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    """
    sample12 = jnp.concatenate([sample1, sample2], axis=0)
    pairwise_dists = jnp.linalg.norm(
        sample12[:, None, :] - sample12[None, :, :], axis=-1)
    pairwise_dists2 = pairwise_dists**2

    kernels = None
    for alpha in alphas:
        if kernels is None:
            kernels = jnp.exp(-alpha * pairwise_dists2)
        else:
            kernels = kernels + jnp.exp(-alpha * pairwise_dists2)

    n1 = sample1.shape[0]
    n2 = sample2.shape[0]
    a00 = 1. / (n1 * (n1 - 1))
    a11 = 1. / (n2 * (n2 - 1))
    a01 = - 1. / (n1 * n2)

    k_1 = kernels[:n1, :n1]
    k_2 = kernels[n1:, n1:]
    k_12 = kernels[:n1, n1:]

    mmd2 = (2 * a01 * jnp.sum(k_12) +
           a00 * (jnp.sum(k_1) - jnp.trace(k_1)) +
           a11 * (jnp.sum(k_2) - jnp.trace(k_2)))

    return mmd2

def test():
    needs_extra_vmap = True
    # logprob = make_two_moons()
    dist = make_circular_mog()
    # dist = make_ring_mixture()
    # dist = make_grid_mog()

    key = jax.random.PRNGKey(0)
    samples = dist.sample(key, n_samples=10_000)
    print(samples.shape)

    import matplotlib.pyplot as plt
    import numpy as np

    lo, hi = -7, 7
    pts = np.linspace(lo, hi, 1000)
    # pts = np.linspace(-4, 4, 1000)
    # pts = np.linspace(-7, 7, 10)
    X, Y = np.meshgrid(pts, pts)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    if needs_extra_vmap:
        Z = np.exp(jax.vmap(jax.vmap(dist.logprob))(pos))
    else:
        Z = np.exp(jax.vmap(dist.logprob)(pos))

    # print(Z.shape)

    f, ax = plt.subplots()

    # """ image plot
    # Z = onp.exp(jax.vmap(jax.vmap(logprob))(pos))
    Z = Z/Z.max()
    # Z = jnp.expand_dims(Z, axis=-1)

    # import matplotlib.cm as cm
    # ax.imshow(Z, extent=(X.min(), X.max(), Y.max(), Y.min()),
    #    interpolation='nearest', cmap=cm.viridis)

    ax.contour(X,Y,Z, alpha=0.5)
    ax.hist2d(samples[:,0], samples[:, 1], bins=100,
              range=((lo, hi), (lo, hi)), cmap=plt.cm.BuPu)


    plt.show()

def test_mmd2():
    dist = make_circular_mog()
    dist2 = make_ring_mixture()

    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)

    samples1 = dist.sample(key1, n_samples=10_000)
    samples2 = dist.sample(key2, n_samples=10_000)
    samples3 = dist2.sample(key3, n_samples=10_000)

    mmd2_same = compute_mmd2(samples1, samples2)
    mmd2_diff = compute_mmd2(samples1, samples3)

    print(f"mmd same = {mmd2_same} vs mmd different = {mmd2_diff}")



if __name__ == '__main__':
    # test()
    test_mmd2()