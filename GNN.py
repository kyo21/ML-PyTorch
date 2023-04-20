import numpy as np
import networkx as nx

class GNN:
    # def __init__():

    
    def build_adj_matrix(self, col):
        G = nx.Graph()

        for i, cols in enumerate(col):
            G.add_nodes_from(
                [(i+1, {'color': cols})]
            )

        for j in range(len(col)):
            if j == 3:
                break
            else:
                G.add_edges_from([(1,j+2)])
        
        A = np.asarray(nx.adjacency_matrix(G).todense())

        return A
        

        





