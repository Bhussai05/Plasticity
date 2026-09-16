import networkx as nx 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import solve_ivp




def pop_rate(t, x, K):
    return x * (1- x/K)





def main():
    K = 1.0
    x_initial = 0.1

    solution = solve_ivp(
        fun=pop_rate,
        t_span=(0.0, 10.0),
        y0=[x_initial],
        args=(K,),
    )

    times = solution.t
    populations = solution.y[0]

    print(f"Initial population: {populations[0]:.4f}")
    print(f"Final population: {populations[-1]:.4f}")


if __name__ == "__main__":
    main()