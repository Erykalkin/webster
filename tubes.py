import numpy as np


def make_random_smooth_tube(L=1000.0, dx=0.1,
                            S0=2.0,
                            amp=0.3,
                            n_harmonics=5,
                            rng=None
                            ):
    if rng is None:
        rng = np.random.default_rng()

    x_nodes = np.arange(0.0, L + dx, dx)
    n_nodes = x_nodes.size

    c = rng.normal(loc=0.0, scale=1.0, size=n_harmonics)
    phi = rng.uniform(0.0, 2 * np.pi, size=n_harmonics)

    noise = np.zeros_like(x_nodes, dtype=float)
    for k in range(1, n_harmonics + 1):
        noise += c[k - 1] * np.sin(2 * np.pi * k * x_nodes / L + phi[k - 1])

    noise /= np.max(np.abs(noise) + 1e-12)

    S = S0 * (1.0 + amp * noise)
    S_min = 0.1 * S0
    S = np.clip(S, S_min, None)

    l = np.full(n_nodes - 1, dx, dtype=float)

    return x_nodes, S, l