import networkx as nx 
import seaborn as sns
import matplotlib.pyplot as plt 


from kuramoto import Kuramoto, plot_activity

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.6)

graph = nx.erdos_renyi_graph(n=100, p =0.5)
matrix = nx.to_numpy_array(graph)

model = Kuramoto(coupling=4, dt=0.01, T=10, n_nodes=len(matrix))

activity = model.run(adj_mat=matrix)

plot_activity(activity)

plt.show()
