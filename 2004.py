import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


rng = np.random.default_rng(7)
N = 5
K = 5
connection = 1

b = 0.1
c = 0.2
epsilon = 0.01


for i in range(N):
    for j in range(i+1, N):
        if rng.random() < connection:

            alpha_0 = rng.random()

            if rng.random() < 0.5:
                predator, prey = 0,1
            else:
                predator, prey = 1,0


autotroph = np.array([True,False])

y_0 = [5,3, alpha_0]


def rates(t,y,c,K_1, epsilon):
    x = y[:N]
    alpha = y[N]

    M = np.zeros((N,N))

    M[predator, prey] = alpha
    M[prey, predator] = -b * alpha
    

    interactions = x * (M@x)

    dx = interactions.copy()

    for i in range(N):
        if autotroph[i]:
            dx[i] += x[i] * (1- x[i]/ K_1)
        else:
            dx[i] += -c*x[i]

        dalpha = epsilon * (x[prey] - x[predator]) * alpha

    return np.concatenate((dx, [dalpha]))

    


sol = solve_ivp(
    rates,
    (0,20),
    y_0,
    args=(c,K, epsilon)
)
plt.plot(sol.t, sol.y[0], label="x1")
plt.plot(sol.t, sol.y[1], label="x2")
plt.plot(sol.t, sol.y[2], label="alpha")

plt.legend()
plt.xlabel("t")
plt.show()