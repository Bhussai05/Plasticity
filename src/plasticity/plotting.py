"""Diagnostics and figures from the original 2004.py simulation."""

import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    sol, config, autotroph, edges, equation, clip_links, generate_all, run_stamp,
):
    """Print the original diagnostics and display or save the original figures."""
    N = config["N"]
    seed = config["seed"]
    n_autotrophs = config["n_autotrophs"]
    connection = config["connection"]
    K = config["K"]
    b = config["b"]
    c = config["c"]
    epsilon = config["epsilon"]
    gamma = config["gamma"]
    EXTINCTION_THRESHOLD = config["EXTINCTION_THRESHOLD"]
    T_END = config["T_END"]
    METHOD = config["METHOD"]
    RTOL = config["RTOL"]
    ATOL = config["ATOL"]
    MAX_STEP = config["MAX_STEP"]
    DESKTOP_PATH = config["DESKTOP_PATH"]

    X = sol.y[:N]
    A_raw_solution = sol.y[N:]

    if not sol.success:
        raise RuntimeError(sol.message)

    if clip_links:
        A_solution = np.clip(A_raw_solution, 0.0, 1.0)
    else:
        A_solution = A_raw_solution

    last_x = X[:, -1]

    laststep = -1
    strengths = []

    for k, (i,j) in enumerate(edges):
        i_alive = X[i, laststep] > EXTINCTION_THRESHOLD
        j_alive = X[j, laststep] > EXTINCTION_THRESHOLD

        if i_alive and j_alive:
            strength = A_solution[k,laststep]

            if strength > 0:
                strengths.append(strength)

    strengths = np.asarray(strengths)

    print("Strengths collected:", strengths.size)

    if strengths.size > 0:
        print("Smallest strength:", strengths.min())
        print("Largest strength:", strengths.max())

        bin_edges = np.geomspace(
            strengths.min(),
            strengths.max(),
            21
        )

        counts,_ = np.histogram(strengths, bins=bin_edges)

        print("counts in each bin", counts)
        print("total counted", counts.sum())

        centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        widths = np.diff(bin_edges)
        number_density = counts / widths

        nonempty = counts > 0
        fig_dist, ax_dist = plt.subplots()

        ax_dist.loglog(
            centres[nonempty],
            number_density[nonempty],
            "o"
        )
        ax_dist.set_xlabel("Link strength A")
        ax_dist.set_ylabel("Number of links per unit strength")
        ax_dist.set_title(f"Equation {equation}: final link distribution")
        ax_dist.grid(alpha=0.25)

    survivors = np.count_nonzero(
        last_x > EXTINCTION_THRESHOLD
    )

    print(f"\nEquation: {equation}")
    print("NumPy clipping:", clip_links)
    print("Solver success:", sol.success)
    print(sol.message)
    print("Last saved time:", sol.t[-1])
    print("Survivors:", survivors, "out of", N)
    print("Population range:", last_x.min(), last_x.max())

    active_raw_A = []
    active_A = []

    for k, (i, j) in enumerate(edges):
        if (
            last_x[i] > EXTINCTION_THRESHOLD
            and last_x[j] > EXTINCTION_THRESHOLD
        ):
            active_raw_A.append(A_raw_solution[k, -1])
            active_A.append(A_solution[k, -1])

    print("Active feeding links:", len(active_A))

    if len(active_A) > 0:
        print(
            "Active raw link range:",
            min(active_raw_A),
            max(active_raw_A)
        )
        print(
            "Active effective link range:",
            min(active_A),
            max(active_A)
        )
    else:
        print("No feeding links remain between surviving species.")

    if clip_links:
        clipping_label = "ON [0, 1]"
        clipping_tag = "clip"
    else:
        clipping_label = "OFF"
        clipping_tag = "noclip"

    if sol.success:
        solver_label = "SUCCESS"
    else:
        solver_label = "FAILED"

    parameters = (
        f"N={N}; autotrophs={n_autotrophs}; "
        f"initial links={len(edges)}; "
        f"connection={connection:g}; seed={seed}\n"
        f"K={K:g}; b={b:g}; c={c:g}; epsilon={epsilon:g}; "
        f"initial populations=U[1, 5]; "
        f"extinction={EXTINCTION_THRESHOLD:g}\n"
        f"Time requested={T_END:g}; "
        f"time reached={sol.t[-1]:.6g}; "
        f"survivors={survivors}/{N}; "
        f"final active links={len(active_A)}\n"
        f"Solver={METHOD}; rtol={RTOL:g}; atol={ATOL:g}; "
        f"max_step={MAX_STEP:g}\n"
        f"Link curves stop when either species becomes extinct."
        f"gamma={gamma:g}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    fig.suptitle(
        f"Eq. ({equation}) | "
        f"np.clip: {clipping_label} | Solver: {solver_label}",
        fontsize=14
    )

    for i in range(N):

        if autotroph[i]:
            species_type = "plant"
        else:
            species_type = "animal"

        axes[0].plot(
            sol.t,
            X[i],
            label=f"x{i + 1} ({species_type})"
        )

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Population")
    axes[0].grid(alpha=0.25)

    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8
    )

    plotted_links_positive = np.all(A_solution > 0.0)

    for k, (i, j) in enumerate(edges):
        axes[1].plot(
            sol.t,
            A_solution[k],
            label=f"{i + 1} eats {j + 1}"
    )

    if len(edges) > 0:

        if plotted_links_positive:
            axes[1].set_yscale("log")
        else:
            axes[1].set_yscale("symlog", linthresh=ATOL)

        axes[1].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8
        )

    else:
        axes[1].text(
            0.5,
            0.5,
            "No feeding links generated",
            transform=axes[1].transAxes,
            ha="center"
        )

    axes[1].set_xlabel("Time")
    axes[1].grid(alpha=0.25)

    if clip_links:
        axes[1].set_ylabel("Effective link strength A")
    else:
        axes[1].set_ylabel("Raw link strength A")

    fig.text(
        0.08,
        0.025,
        parameters,
        fontsize=9,
        va="bottom"
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.76,
        top=0.92,
        bottom=0.18,
        hspace=0.35
    )

    if generate_all:

        filename = (
            f"eq{equation}{clipping_tag}_{run_stamp}.png"
        )

        output_path = DESKTOP_PATH / filename

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        print(f"Saved: {output_path}")

        plt.close(fig)

    else:
        plt.show()
