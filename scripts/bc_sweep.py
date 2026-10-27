from pathlib import Path
import csv 
import sys

import matplotlib.pyplot as plt 
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))

from plasticity.config import load_config
from plasticity.model import initialize_web

B_RANGE = (0.05, 1.0)
C_RANGE = (0.01, 10.0)
GRID_SIZE = 5
REFINE_ROUNDS = 2
POINTS_PER_ROUND = 10
SEARCH_SEED = 123

njit(cache=True)
def fast_rates(
    t, y, c, K, epsilon, equation, clip_links,
    N, b, gamma, threshold, autotroph, edges,
):
    
    x = y[:N].copy()
    for i in range(N):
        if x[i] <= threshold:
            x[i] = 0.0

    raw_A = y[N:]
    A = raw_A.copy()

    if clip_links:
        for k in range(len(A)):
            A[k] = min(1.0, max(0.0, A[k]))

    growth = np.zeros(N)

    for k in range(len(edges)):
        predator, prey = edges[k]
        growth[predator] += b * A[k] * x[prey]
        growth[prey] -= A[k] * x[predator]

    for i in range(N):
        if autotroph[i]:
            growth[i] += 1.0 - x[i] / K
        else:
            growth[i] -= c

    result = np.zeros(len(y))
    result[:N] = x * growth

    for k in range(len(edges)):
        predator, prey = edges[k]

        if x[predator] == 0.0 or x[prey] == 0.0:
            continue

        if equation == 5:
            change = epsilon * (
                (x[prey] - x[predator]) * A[k]
                - gamma * A[k] ** 2
            )
        elif equation == 6:
            change = epsilon * (
                growth[prey] - growth[predator]
            ) * A[k]
        else:  # Equation 7.
            change = epsilon * x[prey] * A[k]

        if clip_links:
            if raw_A[k] >= 1.0 and change > 0.0:
                change = 0.0
            if raw_A[k] <= 0.0 and change < 0.0:
                change = 0.0

        result[N + k] = change

    return result

def main():
    config = load_config()

    autotroph, original_edges, initial = initialize_web(
        config["seed"],
        config["N"],
        config["connection"],
        config["n_autotrophs"],
    )
    edges = np.asarray(original_edges, dtype=np.int64).reshape(-1, 2)

    def solver_args(b, c):
        return (
            c, config["K"], config["epsilon"],
            config["EVOLUTION_EQUATION"], config["CLIP_LINKS"],
            config["N"], b, config["gamma"],
            config["EXTINCTION_THRESHOLD"], autotroph, edges,
        )

    # Compile once before the search.
    fast_rates(0.0, initial, *solver_args(config["b"], config["c"]))

    rng = np.random.default_rng(SEARCH_SEED)

    # Search coordinates: b and log10(c).
    lower = np.array([B_RANGE[0], np.log10(C_RANGE[0])])
    upper = np.array([B_RANGE[1], np.log10(C_RANGE[1])])