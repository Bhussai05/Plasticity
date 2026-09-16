import networkx as nx 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import solve_ivp



state = [1.0, 5]
a = 0.1
b = 0.1
c = 0.01


def pop_rate(t, state, K, a,b,c):

    prey,predator = state

    consume = a * prey * predator

    dprey_dt = (prey * (1- prey/K)) - consume 

    dpred_dt = b * consume - c * predator

    return [dprey_dt, dpred_dt]





def simulate_population(state, K, a,b,c):
   
    return solve_ivp(
        fun=pop_rate,
        t_span=(0.0, 10.0),
        y0=state,
        args=(K,a,b,c),
        t_eval=np.linspace(0.0, 10.0, 201)
    )


def main():
    K = 1.0
    solution = simulate_population(state=state, K=K, a=a, b=b, c=c)

    times = solution.t
    populations_prey = solution.y[0]
    populations_pred = solution.y[1]

    fig, ax = plt.subplots()

    ax.plot(solution.t, solution.y[0], label="prey")
    ax.plot(solution.t, solution.y[1], label="predator")

    ax.axhline(
        K,
        color = "red",
        linestyle = "--",
        label = "Prey carrying capacity"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.legend()

    plt.show()







if __name__ == "__main__":
    main()
