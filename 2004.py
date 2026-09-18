import numpy as np
from scipy.integrate import solve_ivp

x_lim = 5
y_0 = [5,3,2]

alpha = 1
b = 0.1
c = 0.2


def rates(t,y, x_lim, alpha, b, c):

    x = np.array(y)

    M =  np.array([
        [0.0, - b * alpha, 0.0],
        [alpha, 0.0, -b*alpha ],
        [0.0, alpha, 0.0]
        ])

    interaction = x * (M @ x )

    x1, x2, x3 = x


    dx1 = x1 *(1 - x1 / x_lim) + interaction[0]
    dx2 = (-1) * c *x2 + interaction[1]
    dx3 = (-1) * c *x3 + interaction[2]

    return[dx1, dx2, dx3]



sol = solve_ivp(
    rates,
    (0,20),
    y_0,
    args=(x_lim, alpha, b,c )
)



