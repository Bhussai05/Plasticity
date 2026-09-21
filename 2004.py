import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


rng = np.random.default_rng(7)
N = 5
K = 5
connection = 0.5

b = 0.1
c = 0.2
epsilon = 0.01

n_autotrophs = 2

autotroph = np.zeros(N, dtype=bool)

autotroph_indices = rng.choice(
    N,
    size=n_autotrophs,
    replace=False
)

autotroph[autotroph_indices] = True

print("Autotropph Mask:")
print(autotroph)


edges = []
A0 = []


for i in range(N):
    for j in range(i+1, N):

        if rng.random() < connection:

            alpha_0 = rng.random()

            if rng.random() < 0.5:
                predator, prey = i,j
            else:
                predator, prey = j,i


            edges.append((predator, prey))
            A0.append(alpha_0)

A0 = np.array(A0, dtype=float)

print("\nEdges:")
for k, (predator, prey) in enumerate(edges):
    print(
        f"link {k}: species {predator + 1} eats "
        f"species {prey + 1}, A0 = {A0[k]:.3f}"
    )

x_0 = np.array([5,3,2,4,1], dtype=float)



y_0 = np.concatenate((x_0, A0))


print("\nInitial full state:")
print(y_0)

print("\nInitial populations:")
print(y_0[:N])

print("\nInitial link strengths:")
print(y_0[N:])





def rates(t,y,c,K_1, epsilon):
    x = y[:N]
    A = y[N:]

    M = np.zeros((N,N))

    for k, (predator,prey) in enumerate(edges):

        alpha = min(A[k], 1.0)

        M[predator,prey] = alpha

        M[prey,predator] = -b * alpha
    

    interactions = x * (M@x)

    dx = interactions.copy()

    for i in range(N):
        if autotroph[i]:
            dx[i] += x[i] * (1- x[i]/ K_1)
        else:
            dx[i] += -c*x[i]


    dA = np.zeros(len(A))

    for k, (predator,prey) in enumerate(edges):

        alpha = min(A[k],1.0)

        raw_dA = (
            epsilon * x[prey] - x[predator] * alpha
        )

        if A[k] >= 1.0 and raw_dA > 0:
            raw_dA = 0.0

        dA[k] = raw_dA

    return np.concatenate((dx, dA))

t_eval = np.linspace(0,20, 1000)




sol = solve_ivp(
    rates,
    (0,20),
    y_0,
    args=(c,K, epsilon),
    t_eval=t_eval,

    max_step = 0.05
)

print("\nSolver Sucess:")
print(sol.success)
print(sol.message)



plt.plot(sol.t, sol.y[0], label="x1")
plt.plot(sol.t, sol.y[1], label="x2")
plt.plot(sol.t, sol.y[2], label="alpha")

plt.legend()
plt.xlabel("t")
plt.show()