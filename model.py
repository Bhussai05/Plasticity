import networkx as nx 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np





def pop_rate(x, K):
    return x * (1- x/K)


K = 1.0
for x in [0.1, 1.0, 1.5]:
    rate = pop_rate(x,K)
    print()
