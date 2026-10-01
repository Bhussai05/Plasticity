import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


EVOLUTION_EQUATION = 6
CLIP_LINKS = True

seed = 7
N = 15
K = 5
connection = 0.15
b = 0.1
c = 0.01
epsilon = 0.01

n_autotrophs = 5
EXTINCTION_THRESHOLD = 1e-6

T_END = 1000.0
METHOD = "DOP853"
RTOL = 1e-6
ATOL = 1e-9
MAX_STEP = np.inf

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
    for j in range(i+1, N):
        if autotroph[i] and autotroph[j]:
            continue
        if rng.random() < connection:
            alpha_0 = rng.random()
            if autotroph[i]:
                predator,prey = j,i
            elif autotroph[j]:
                predator, prey = i,j
            elif rng.random() < 0.5:
                predator, prey = i,j
            else:
                predator,prey = j,i 


            edges.append((predator, prey))
            A0.append(alpha_0)

A0 = np.array(A0, dtype=float)

x_0 = rng.uniform(1.0,5.0, size=N)

y_0 = np.concatenate((x_0, A0))



def rates(t,y,c,K_1, epsilon,equation, clip_links):

    x = y[:N].copy()

    for i in range(N):
        if x[i] <= EXTINCTION_THRESHOLD:
            x[i] = 0.0

    A_raw= y[N:]

    if clip_links:
        A = np.clip(A_raw, 0.0, 1.0)
    else:
        A = A_raw

    M = np.zeros((N,N))

    for k, (predator,prey) in enumerate(edges):

        M[predator,prey] = b*A[k]

        M[prey,predator] =  -A[k]
    

    relative_growth= M@x

    for i in range(N):
        if autotroph[i]:
            relative_growth[i] += 1.0 *  x[i]/ K_1
        else:
            relative_growth[i] -= -c

    dx = x * relative_growth

    dA = np.zeros(len(A))

    for k, (predator,prey) in enumerate(edges):

        if equation == 5:

            dA[k] = epsilon * (x[prey] - x[predator]) * A[k]

        elif equation == 6:

            dA[k] = (epsilon *(relative_growth[prey] - relative_growth[predator]) * A[k])

        elif equation == 7:

            dA[k] = epsilon *x[prey] * A[k]

        if clip_links:

            if A_raw[k] >= 1.0 and dA[k] > 0.0:
                dA[k] = 0.0
            if A_raw[k] <= 0.0 and dA[k] < 0.0:
                dA[k] = 0.0

    return np.concatenate((dx, dA))





sol = solve_ivp(
    rates,
    (0,T_END),
    y_0,
    args=(c,K, epsilon,EVOLUTION_EQUATION, EXTINCTION_THRESHOLD, CLIP_LINKS),
    method=METHOD,
    rtol=RTOL,
    atol=ATOL,
    max_step=MAX_STEP
)

print("\nSolver Sucess:")
print(sol.success)
print(sol.message)

X = sol.y[:N]
A_raw_solution = sol.y[N:]

if CLIP_LINKS:
    A_solution = np.clip(A_raw_solution, 0.0,1.0)
else:
    A_solution = A_raw_solution

last_x = X[:,-1]

survivors = np.count_nonzero(
    last_x > EXTINCTION_THRESHOLD
)

print("Equation:", EVOLUTION_EQUATION)
print("NumPy clipping:", CLIP_LINKS)
print("Solver success:", sol.success)
print(sol.message)
print("Last saved time:", sol.t[-1])
print("Survivors:", survivors, "out of", N)
print("Population range:", last_x.min(), last_x.max())

if len(edges) > 0:
    last_raw_A = A_raw_solution[:, -1]
    last_A = A_solution[:,-1]

print("Raw link range:", last_raw_A.min(), last_raw_A.max())
print("Effective link range:m", last_A.min(), last_A.max())


if CLIP_LINKS:
    clipping_label = "ON [0,1]"
else:
    clipping_label = "OFF"

if sol.success:
    solver_label = "SUCCESS"
else:
    solver_label = "FAILED"

parameters = (
    f"N={N}; autotrophs={n_autotrophs}; links={len(edges)}; "
    f"connection={connection:g}; seed={seed}\n"
    f"K={K:g}; b={b:g}; c={c:g}; epsilon={epsilon:g}; "
    f"initial populations=U[1, 5]; extinction={EXTINCTION_THRESHOLD:g}\n"
    f"Time requested={T_END:g}; time reached={sol.t[-1]:.6g}; "
    f"survivors={survivors}/{N}\n"
    f"Solver={METHOD}; rtol={RTOL:g}; atol={ATOL:g}; "
    f"max_step={MAX_STEP:g}"
)

fig, axes = plt.subplots(2,1,figsize=(12,10))

fig.suptitle(
    f"Eq. ({EVOLUTION_EQUATION}) |",
    f"np.clip: {clipping_label} | Solver: {solver_label}",
    fontsize = 14
)


for i in range(N):
    if autotroph[i]:
        species_type = "plant"
    else:
        species_type = "animal"

    axes[0].plot(
        sol.t,
        X[i],
        label=f"x{i+1} ({species_type})"
    )

axes[0].set_xlabel("Time")
axes[0].set_ylabel("Population")
axes[0].grid(alpha=0.25)
axes[0].legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=8
)


for k, (predator,prey) in enumerate(edges):
    axes[1].plot(
        sol.t,
        A_solution[k],
        label = f"{predator +1} eats {prey+1}"
    )

if len(edges) > 0:

    if np.all(A_solution > 0.0):
        axes[1].set_yscale("log")
    else:
        axes[1].set_yscale("symlog", linthresh = ATOL)

    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(1.02,1.0),
        fontsize=8
    )

axes[1].set_xlabel("Time")

if CLIP_LINKS:
    axes[1].set_ylabel("Effective link strength A")
else:
    axes[2].set_ylabel("Raw link strength A")

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

plt.show()

