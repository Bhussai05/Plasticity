import numpy as np
from scipy.integrate import solve_ivp



rng = np.random.default_rng(7)
N = 2
K = 5
y_0 = [5,3]
connection = 1

b = 0.1
c = 0.2

M = np.zeros((N,N))

for i in range(N):
    for j in range(i+1, N):
        if rng.random() < connection:

            alpha = rng.random()

            if rng.random() < 0.5:
                predator, prey = i,j
            else:
                predator, prey = j,i

            M[predator, prey] = alpha
            M[prey, predator] = -b * alpha

autotroph = np.array([True,False])


def rates(t,y,c,K_1):
    x = np.array(y)

    interactions = x * (M@x)

    dx = interactions.copy()

    for i in range(N):
        if autotroph[i]:
            dx[i] += x[i] * (1- x[i]/ K_1)
        else:
            dx[i] += -c*x[i]

    return dx

    


sol = solve_ivp(
    rates,
    (0,20),
    y_0,
    args=(c,K)
)



