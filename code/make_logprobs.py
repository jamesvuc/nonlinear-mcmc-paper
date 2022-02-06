
import jax
import jax.numpy as jnp

def make_mog():
    from jax.scipy.stats import multivariate_normal as mvn
    mus = jnp.array([[0.0, 0.0],
                    [2.5,  4.0],
                    [4.0,  0.0]])

    sigmas = jnp.array([ [[1.0, 0.0],
                         [0.0, 2.0]],
                        
                        [[2.0, -1.0],
                         [-1.0, 1.0]],

                        [[1.0, 0.1],
                         [0.1, 2.0]] ])

    # @jax.jit
    def logprob(z):
        return jnp.log(mvn.pdf(z, mean=mus[0], cov=sigmas[0]) + \
                      mvn.pdf(z, mean=mus[1], cov=sigmas[1]) + \
                      mvn.pdf(z, mean=mus[2], cov=sigmas[2]))

    return logprob

def make_egg_carton_logprob():
    pass


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

    return logprob 

def make_circular_mog(n_modes=8):
    pi = 3.141592658
    scale = 2 / 3 * jnp.sin(pi / n_modes)

    def logprob(z):
        components = []
        for i in range(n_modes):
            c = ((z[0] - 2 * jnp.sin(2 * pi / n_modes * i)) ** 2
                  + (z[1] - 2 * jnp.cos(2 * pi / n_modes * i)) ** 2)\
                 / (2 * scale ** 2)
            components.append(c)

        log_p = - jnp.log(2 * pi * scale ** 2 * n_modes) \
                + jax.scipy.special.logsumexp(-jnp.array(components))
        return log_p

    return logprob

def make_ring_mixture(n_rings=2):
    n_rings = n_rings
    scale = 1 / 4 / n_rings

    def logprob(z):
        components = []
        for i in range(n_rings):
            c = ((_norm(z) - 2 / n_rings * (i + 1)) ** 2) \
                 / (2 * scale ** 2)
            components.append(c)
        return jax.scipy.special.logsumexp(-jnp.array(components))

    return logprob

def test():
    # logprob = make_mog()
    # needs_extra_vmap = False

    needs_extra_vmap = True
    # logprob = make_two_moons()
    # logprob = make_circular_mog()
    logprob = make_ring_mixture()


    import numpy as np
    from matplotlib import pyplot as plt
    # pts = np.linspace(-7, 7, 1000)
    pts = np.linspace(-4, 4, 1000)
    # pts = np.linspace(-7, 7, 10)
    X, Y = np.meshgrid(pts, pts)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    if needs_extra_vmap:
        Z = np.exp(jax.vmap(jax.vmap(logprob))(pos))
    else:
        Z = np.exp(jax.vmap(logprob)(pos))
    
    f, ax = plt.subplots()
    ax.contour(X, Y, Z,)
    # ax.set_xlim(-2, 6)
    # ax.set_ylim(-5, 7)
    plt.show()

if __name__ == '__main__':
    test()