import networkx as nx 
import seaborn as sns

from kuramoto import Kuramoto, plot_activity

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.6)

graph = nx.erdos_renyi_graph(n=100, p =0.5)
matrix = nx.to_numpy_array(graph)

model = 