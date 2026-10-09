import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime




EVOLUTION_EQUATION = 6
CLIP_LINKS = False

seed = 7
N = 15
K = 5
connection = 0.7

b = 0.1
c = 0.01
epsilon = 0.1

n_autotrophs = 5
EXTINCTION_THRESHOLD = 1e-6

T_END = 100000.0
METHOD = "DOP853"
RTOL = 1e-6
ATOL = 1e-9
MAX_STEP = np.inf

DESKTOP_PATH = Path("/Users/bilalhussain/Desktop")

if EVOLUTION_EQUATION not in (5, 6, 7):
    raise ValueError("EVOLUTION_EQUATION must be 5, 6, or 7.")



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

for i in range(N):
    for j in range(i + 1, N):

        if autotroph[i] and autotroph[j]:
            continue

        if rng.random() < connection:
            alpha_0 = rng.random()

            if autotroph[i]:
                predator, prey = j, i

            elif autotroph[j]:
                predator, prey = i, j

            elif rng.random() < 0.5:
                predator, prey = i, j

            else:
                predator, prey = j, i

            edges.append((predator, prey))
            A0.append(alpha_0)

A0 = np.array(A0, dtype=float)

x_0 = rng.uniform(1.0, 5.0, size=N)

y_0 = np.concatenate((x_0, A0))




def rates(t, y, c, K_1, epsilon, equation, clip_links):

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

    for k, (predator, prey) in enumerate(edges):
        M[predator, prey] = b * A[k]
        M[prey, predator] = -A[k]

    relative_growth = M @ x

    for i in range(N):
        if autotroph[i]:
            relative_growth[i] += 1.0 - x[i] / K_1
        else:
            relative_growth[i] -= c

    dx = x * relative_growth

    dA = np.zeros(len(A))

    for k, (predator, prey) in enumerate(edges):

        # Freeze the link if either species is extinct.
        if x[predator] == 0.0 or x[prey] == 0.0:
            continue

        if equation == 5:
            dA[k] = (
                epsilon
                * (x[prey] - x[predator])
                * A[k]
            )

        elif equation == 6:
            dA[k] = (
                epsilon
                * (
                    relative_growth[prey]
                    - relative_growth[predator]
                )
                * A[k]
            )

        elif equation == 7:
            dA[k] = epsilon * x[prey] * A[k]

        if clip_links:

            if A_raw[k] >= 1.0 and dA[k] > 0.0:
                dA[k] = 0.0

            if A_raw[k] <= 0.0 and dA[k] < 0.0:
                dA[k] = 0.0

   
    return np.concatenate((dx, dA))




answer = input(
    "Generate all 6 figures and save them to Desktop? [y/N]: "
)

generate_all = answer.strip().lower() in ("y", "yes")

run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if generate_all:
    runs = [
        (5, True),
        (5, False),
        (6, True),
        (6, False),
        (7, True),
        (7, False)
    ]

    DESKTOP_PATH.mkdir(parents=True, exist_ok=True)

else:
    runs = [(EVOLUTION_EQUATION, CLIP_LINKS)]




for equation, clip_links in runs:

    sol = solve_ivp(
        rates,
        (0.0, T_END),
        y_0.copy(),
        args=(c, K, epsilon, equation, clip_links),
        method=METHOD,
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP
    )

    X = sol.y[:N]
    A_raw_solution = sol.y[N:]


    if not sol.success:
        raise RuntimeError(sol.message)



    if clip_links:
        A_solution = np.clip(A_raw_solution, 0.0, 1.0)
    else:
        A_solution = A_raw_solution

    last_x = X[:, -1]

    j = -1
    strengths = []


    for k, (predator,prey) in enumerate(edges):
        predator_alive = X[predator, j] > EXTINCTION_THRESHOLD
        prey_alive = X[prey, j] > EXTINCTION_THRESHOLD

        if predator_alive and prey_alive:
            strength = A_solution[k,j]

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
            11
        )

        counts,_ = np.histogram(strengths, bins=bin_edges)

        print("counts in each bin", counts)
        print("total counted", counts.sum())

        centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        widths = np.diff(bin_edges)
        number_density = counts/widths 

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

    for k, (predator, prey) in enumerate(edges):
        if (
            last_x[predator] > EXTINCTION_THRESHOLD
            and last_x[prey] > EXTINCTION_THRESHOLD
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

    for k, (predator, prey) in enumerate(edges):
        axes[1].plot(
            sol.t,
            A_solution[k],
            label=f"{predator + 1} eats {prey + 1}"
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