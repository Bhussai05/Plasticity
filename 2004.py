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
    (0,500),
    y_0,
    args=(c,K, epsilon),
    # t_eval=t_eval,
    # max_step = 0.05
)

print("\nSolver Sucess:")
print(sol.success)
print(sol.message)

X = sol.y[:N]

A_solution = sol.y[N:]


plt.figure()

for i in range(N):
    plt.plot(
        sol.t,
        X[i],
        label = f"x{i+1}"
    )

plt.xlabel("Time")
plt.ylabel("population")
plt.legend()
plt.show()

plt.figure()

for k in range(len(edges)):
    predator, prey = edges[k]

    plt.plot(
        sol.t,
        A_solution[k],
        label = f"{predator +1} eats {prey+1}"

    )

plt.xlabel("Time")
plt.ylabel("Link strenghts A")
plt.yscale("log")
plt.legend()
plt.show()


