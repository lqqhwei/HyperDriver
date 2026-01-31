import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix

class DataLoader:
    def __init__(self, static_dir):
        """
        [Optimized] Removed label_dir argument as Node_Labels are no longer needed.
        """
        self.static_dir = static_dir
        self.static_file = os.path.join(static_dir, 'Static_PPIN.txt')
        
        # Node mapping table: Protein Name -> Index
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.num_nodes = 0
        
    def load_static_graph(self):
        """
        Load a static PPI network and return a NetworkX object and an adjacency matrix.
        """
        print(f"Loading Static PPIN from {self.static_file}...")
        
        # 1. Read the edge list and build a node set.
        edges = []
        nodes = set()
        
        # Check if file exists
        if not os.path.exists(self.static_file):
             raise FileNotFoundError(f"Static PPIN file not found: {self.static_file}")

        with open(self.static_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                u, v = parts[0], parts[1]
                # Compatible with possible weighted columns; defaults to 1.0.
                w = float(parts[2]) if len(parts) > 2 else 1.0
                
                edges.append((u, v, w))
                nodes.add(u)
                nodes.add(v)
        
        # 2. Establish index mapping
        sorted_nodes = sorted(list(nodes))
        self.node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(sorted_nodes)}
        self.num_nodes = len(sorted_nodes)
        
        print(f"Graph loaded. Nodes: {self.num_nodes}, Edges: {len(edges)}")
        
        # 3. Building a NetworkX Graph
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
            
        return G

    def get_subgraph_data(self, G, node_indices):
        """
        Auxiliary function: Data structure for extracting subgraphs.
        Kept unchanged as it relies only on G and indices.
        """
        subgraph_nodes = [self.idx_to_node[i] for i in node_indices]
        G_sub = G.subgraph(subgraph_nodes).copy()
        
        # Remap the subgraph indices from 0 to M.
        sub_nodes_list = sorted(list(G_sub.nodes()))
        sub_node_to_idx = {n: i for i, n in enumerate(sub_nodes_list)}
        
        # Extracting the adjacency matrix A_sub from the subgraph
        A_sub = nx.to_numpy_array(G_sub, nodelist=sub_nodes_list)
        
        return G_sub, A_sub, sub_node_to_idx, sub_nodes_list

# Test code
if __name__ == "__main__":
    # Example usage updated
    # label_dir argument is removed
    loader = DataLoader("./data") 
    G = loader.load_static_graph()