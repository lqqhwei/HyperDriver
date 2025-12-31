import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix

class DataLoader:
    def __init__(self, static_dir,label_dir):
        self.static_dir = static_dir
        self.label_dir = label_dir
        self.static_file = os.path.join(static_dir, 'Static_PPIN.txt')
        self.label_file = os.path.join(label_dir, 'Node_Labels_with_essential.csv')
        
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
        
        with open(self.static_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                u, v = parts[0], parts[1]
                # Compatible with possible weighted columns; defaults to 1.0 if none are specified.
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

    def load_essential_labels(self):
        """
        Load the required protein tags and return a binary list (in idx order).
        """
        print(f"Loading Labels from {self.label_file}...")
        df = pd.read_csv(self.label_file)
        
        # Assuming the CSV column names are: 'Protein', 'Essential' (or others, depending on the actual file)
        # Here's a simple error handling mechanism: print the column names for debugging purposes.
        # print("Columns:", df.columns)
        
        # Create the necessary protein set
        # Filter by setting Essential == 'E' (assuming E represents Essential, adjust according to actual data).
        # Here, the Essential column is usually 'E' (Essential) or 'NE' (Non-Essential).
        essential_nodes = set(df[df['Essential'] == 'E']['Protein'].values)
        
        labels = np.zeros(self.num_nodes, dtype=int)
        count = 0
        for i in range(self.num_nodes):
            node_name = self.idx_to_node[i]
            if node_name in essential_nodes:
                labels[i] = 1
                count += 1
                
        print(f"Essential proteins mapped: {count}/{self.num_nodes}")
        return labels

    def get_subgraph_data(self, G, node_indices):
        """
        Auxiliary function: Data structure for extracting subgraphs from a list of node indices.
        For Case Study
        """
        subgraph_nodes = [self.idx_to_node[i] for i in node_indices]
        G_sub = G.subgraph(subgraph_nodes).copy()
        
        # Remap the subgraph indices from 0 to M.
        sub_nodes_list = sorted(list(G_sub.nodes()))
        sub_node_to_idx = {n: i for i, n in enumerate(sub_nodes_list)}
        
        # Extracting the adjacency matrix A_sub from the subgraph
        A_sub = nx.to_numpy_array(G_sub, nodelist=sub_nodes_list)
        
        return G_sub, A_sub, sub_node_to_idx, sub_nodes_list

# Test code (can be commented out at runtime)
if __name__ == "__main__":
    # Assuming your data is in current_dir/data
    loader = DataLoader("./data")
    G = loader.load_static_graph()
    labels = loader.load_essential_labels()