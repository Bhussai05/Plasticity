"""Food-web initialization and dynamics with fixed environmental competition."""

import numpy as np


def initialize_web(seed, N, connection, n_autotrophs,b):
    """Return biological species, feeding links, state, and fixed environment links."""
    rng = np.random.default_rng(seed=seed)

    autotroph = np.zeros(N, dtype=bool)

    autotroph_indices = rng.choice(
        N,
        size=n_autotrophs,
        replace=False
    )

    autotroph[autotroph_indices] = True

    edges = []
    A0 = []

    for u in range(N):
        for v in range(u + 1, N):

            if autotroph[u] and autotroph[v]:
                continue

            if rng.random() < connection:
                alpha_0 = rng.random()

                if autotroph[u]:
                    i, j = v, u

                elif autotroph[v]:
                    i, j = u, v

                elif rng.random() < 0.5:
                    i,j = u, v

                else:
                    i,j = v, u

                edges.append((i, j))
                A0.append(alpha_0)

    A0 = np.array(A0, dtype=float)

    x_0 = np.where(autotroph,1.0, b)

    y_0 = np.concatenate((x_0, A0))

    # Draw these last to preserve the biological network's random draw order.
    env_strengths = np.zeros(N)
    env_strengths[autotroph] = np.abs(
        1.0 - 2.0 * rng.random(np.count_nonzero(autotroph))
    )

    return autotroph, edges, y_0, env_strengths


def rates(
    t, y, c, H, epsilon, equation, clip_links,
    N, b, gamma, EXTINCTION_THRESHOLD, autotroph, edges, env_strengths,
):
    """Compute population and biological link derivatives with a fixed environment."""

    x = y[:N].copy()

    for i in range(N):
        if x[i] <= EXTINCTION_THRESHOLD:
            x[i] = 0.0

    A_raw = y[N:]

    if clip_links:
        A = np.clip(A_raw, 0.0, 1.0)
    else:
        A = A_raw

    M = np.zeros((N, N))

    for k, (i, j) in enumerate(edges):
        M[i, j] = b * A[k]
        M[j, i] = -A[k]

    relative_growth = M @ x

    env_pressure = np.dot(env_strengths, x)

    for i in range(N):
        if autotroph[i]:
            relative_growth[i] += (
                1.0 - c - x[i] / H
                - (env_pressure - env_strengths[i] * x[i]) / H
            )
        else:
            relative_growth[i] -= c

    dx = x * relative_growth

    L = env_strengths.copy()
    for k, (i, j) in enumerate(edges):
        L[i] += b * A[k]
        L[j] += b * A[k]    

    dA = np.zeros(len(A))

    for k, (i, j) in enumerate(edges):

        # Freeze the link if either species is extinct.
        if x[i] == 0.0 or x[j] == 0.0:
            continue

        if equation == 5:
            dA[k] = epsilon * (
                 (x[j] - x[i])
                * A[k] - gamma * A[k]**2
            )

        elif equation == 6:
            dA[k] = (
                epsilon
                * (
                    relative_growth[j]
                    - relative_growth[i]
                )
                * A[k]
            )

        elif equation == 7:
            dA[k] = epsilon * x[j] * A[k]

        elif equation == 8:
            dA[k] = epsilon * A[k] * (
            x[j] / L[j] - x[i] / L[i]
            )

        if clip_links:

            if A_raw[k] >= 1.0 and dA[k] > 0.0:
                dA[k] = 0.0

            if A_raw[k] <= 0.0 and dA[k] < 0.0:
                dA[k] = 0.0

    return np.concatenate((dx, dA))
