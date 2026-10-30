from pathlib import Path
import sys

import matplotlib.pyplot as plt 
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))

from plasticity.config import load_config
from plasticity.model import initialize_web

B_RANGE = (0.01, 1.0)
C_RANGE = (0.01, 10.0)
GRID_SIZE = 5
REFINE_ROUNDS = 2
POINTS_PER_ROUND = 10
SEARCH_SEED = 123

@njit(cache=True)
def fast_rates(
    t, y, c, H, epsilon, equation, clip_links,
    N, b, gamma, threshold, autotroph, edges, env_strengths,
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

    env_pressure = np.dot(env_strengths, x)

    for i in range(N):
        if autotroph[i]:
            growth[i] += (
                1.0- c- x[i] / H- (env_pressure - env_strengths[i] * x[i] ) / H
        )
        else:
            growth[i] -= c

    result = np.zeros(len(y))
    result[:N] = x * growth

    L = env_strengths.copy()
    for k in range(len(edges)):
        predator, prey = edges[k]
        L[predator] += b * A[k]
        L[prey] += b * A[k]

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
        elif equation == 7: # Equation 7.
            change = epsilon * x[prey] * A[k]
        elif equation == 8:
            change = epsilon * A[k] * (
                x[prey] / L[prey] - x[predator] / L[predator]
            )

        if clip_links:
            if raw_A[k] >= 1.0 and change > 0.0:
                change = 0.0
            if raw_A[k] <= 0.0 and change < 0.0:
                change = 0.0

        result[N + k] = change

    return result

def main():
    config = load_config()

    autotroph, original_edges, initial, env_strengths = initialize_web(
        config["seed"],
        config["N"],
        config["connection"],
        config["n_autotrophs"],
        config["b"],
    )
    edges = np.asarray(original_edges, dtype=np.int64).reshape(-1, 2)

    def solver_args(b, c):
        return (
            c, config["H"], config["epsilon"],
            config["EVOLUTION_EQUATION"], config["CLIP_LINKS"],
            config["N"], b, config["gamma"],
            config["EXTINCTION_THRESHOLD"], autotroph, edges, env_strengths,
        )

    # Compile once before the search.
    fast_rates(0.0, initial, *solver_args(config["b"], config["c"]))

    rng = np.random.default_rng(SEARCH_SEED)

    # Search coordinates: b and log10(c).
    lower = np.array([B_RANGE[0], np.log10(C_RANGE[0])])
    upper = np.array([B_RANGE[1], np.log10(C_RANGE[1])])

    results = []
    seen = set()

    def evaluate(point):
        b, log_c = point
        c = 10.0 ** log_c

        key = tuple(np.round(point, 12))
        if key in seen:
            return
        seen.add(key)

        print(
            f"Run {len(seen)}: b={b:.5g}, c={c:.5g}",
            flush=True,
        )

        candidate_initial = initial.copy()
        candidate_initial[:config["N"]] = np.where(autotroph, 1.0, b)


        try:
            sol = solve_ivp(
                fast_rates,
                (0.0, config["T_END"]),
                candidate_initial,
                args=solver_args(b, c),
                method=config["METHOD"],
                rtol=config["RTOL"],
                atol=config["ATOL"],
                max_step=config["MAX_STEP"],
                t_eval=[config["T_END"]],
            )

            if not sol.success:
                print("  Failed:", sol.message)
                return

            if not np.isfinite(sol.y).all():
                print("  Failed: nonfinite final state")
                return

        except (ValueError, RuntimeError, FloatingPointError) as exc:
            print("  Failed:", exc)
            return

        alive = (
            sol.y[:config["N"], -1]
            > config["EXTINCTION_THRESHOLD"]
        )

        plants = int(np.count_nonzero(alive & autotroph))
        animals = int(np.count_nonzero(alive & ~autotroph))

        results.append({
            "b": float(b),
            "c": float(c),
            "survivors": plants + animals,
            "plants": plants,
            "animals": animals,
        })

        print(
            f"  Survivors: {plants + animals} "
            f"({plants} plants + {animals} animals)"
        )

    # First scan: evenly spaced b and logarithmically spaced c.
    for b in np.linspace(lower[0], upper[0], GRID_SIZE):
        for log_c in np.linspace(lower[1], upper[1], GRID_SIZE):
            evaluate(np.array([b, log_c]))

    # Also evaluate your current YAML parameters if inside the bounds.
    if config["c"] > 0:
        current = np.array([
            config["b"], np.log10(config["c"])
        ])
        if np.all(current >= lower) and np.all(current <= upper):
            evaluate(current)

    # Refine near the best sampled regions.
    for round_index in range(REFINE_ROUNDS):
        if not results:
            break

        candidates = results.copy()
        rng.shuffle(candidates)

        leaders = sorted(
            candidates,
            key=lambda row: row["survivors"],
            reverse=True,
        )[:5]

        radius = (
            (upper - lower)
            / (GRID_SIZE - 1)
            / (2 ** round_index)
        )

        for j in range(POINTS_PER_ROUND):
            if j % 4 == 0:
                # Keep exploring outside the leading regions.
                point = rng.uniform(lower, upper)
            else:
                leader = leaders[j % len(leaders)]
                center = np.array([
                    leader["b"], np.log10(leader["c"])
                ])

                point = rng.uniform(
                    np.maximum(lower, center - radius),
                    np.minimum(upper, center + radius),
                )

            evaluate(point)

    if not results:
        print("No successful simulations.")
        return

    maximum = max(row["survivors"] for row in results)
    best = [
        row for row in results
        if row["survivors"] == maximum
    ]

    print(f"\nBest sampled result: {maximum}/{config['N']} survivors")
    for row in best:
        print(
            f"b={row['b']:.6g}, c={row['c']:.6g}: "
            f"{row['plants']} plants + {row['animals']} animals"
        )

    # Colour the background using the nearest successful sample.
    coordinates = np.array([
        [row["b"], np.log10(row["c"])]
        for row in results
    ])
    scores = np.array([
        row["survivors"] for row in results
    ])

    grid_b, grid_log_c = np.meshgrid(
        np.linspace(lower[0], upper[0], 200),
        np.linspace(lower[1], upper[1], 200),
    )

    estimated = griddata(
        coordinates,
        scores,
        (grid_b, grid_log_c),
        method="nearest",
        rescale=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6), constrained_layout=True
    )

    image = ax.pcolormesh(
        grid_b,
        10.0 ** grid_log_c,
        estimated,
        shading="auto",
        cmap="Blues",
        vmin=0,
        vmax=config["N"],
    )
    fig.colorbar(image, ax=ax, label="Total surviving species")

    ax.scatter(
        coordinates[:, 0],
        10.0 ** coordinates[:, 1],
        color="black",
        s=15,
        label="Simulated pairs",
    )
    ax.scatter(
        [row["b"] for row in best],
        [row["c"] for row in best],
        color="red",
        marker="*",
        s=130,
        label="Best sampled pairs",
    )

    ax.set_xlabel("Conversion efficiency b")
    ax.set_ylabel("Animal mortality c")
    ax.set_yscale("log")
    ax.set_title(
        f"Best sampled: {maximum}/{config['N']} survivors "
        f"at t={config['T_END']:g}\n"
        "Background colours estimate the nearest sample"
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
