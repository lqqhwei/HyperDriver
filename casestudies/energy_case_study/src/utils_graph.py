import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

class SubgraphSelector:
    def __init__(self, G, target_size=(50, 150)):
        """
        G: NetworkX graph (Static PPIN)
        target_size: tuple (min_nodes, max_nodes) Expected subgraph size
        """
        self.G = G
        self.min_size = target_size[0]
        self.max_size = target_size[1]

    def get_best_subgraph(self):
        """
        Try to find the subgraph that is best suited for Control Energy experiments.
        Priority strategy: High-density communities.
        Alternative strategy: Hub-centric local neighborhood (Hub-BFS).
        """
        print(f"Searching for optimal subgraph (Target size: {self.min_size}-{self.max_size})...")
        
        # Strategy 1: Try community discovery
        best_community = self._find_best_community()
        if best_community:
            print(f"Method: Community Detection. Selected cluster size: {len(best_community)}")
            return self._extract_subgraph_from_nodes(best_community)
        
        # Strategy 2: If a community of suitable size cannot be found, use a Hub extension.
        print("No perfect community found. Switching to Hub-Expansion strategy.")
        best_hub_region = self._grow_from_hub()
        print(f"Method: Hub Expansion. Selected region size: {len(best_hub_region)}")
        return self._extract_subgraph_from_nodes(best_hub_region)

    def _find_best_community(self):
        """
        Using a greedy modularity maximization algorithm to find communities
        """
        try:
            # Finding a community (this is a list of node sets)
            communities = list(greedy_modularity_communities(self.G))
            
            candidates = []
            for c in communities:
                if self.min_size <= len(c) <= self.max_size:
                    candidates.append(list(c))
            
            if not candidates:
                return None
            
            # Among the candidates, the community with the highest density was selected.
            # Higher density implies more complex control relationships, better showcasing the advantages of the minimum energy algorithm.
            best_c = None
            max_density = -1.0
            
            for c_nodes in candidates:
                subg = self.G.subgraph(c_nodes)
                density = nx.density(subg)
                if density > max_density:
                    max_density = density
                    best_c = c_nodes
            
            return best_c

        except Exception as e:
            print(f"Community detection failed: {e}")
            return None

    def _grow_from_hub(self):
        """
        Start using BFS from the node with the highest degree and continue until the maximum size is reached.
        """
        # Find the node with the largest degree
        degree_dict = dict(self.G.degree())
        sorted_nodes = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)
        top_hub = sorted_nodes[0][0]
        
        selected_nodes = {top_hub}
        queue = [top_hub]
        
        while len(selected_nodes) < self.max_size and queue:
            current = queue.pop(0)
            neighbors = list(self.G.neighbors(current))
            
            # Sort by neighbor's degree, prioritizing important neighbors.
            neighbors.sort(key=lambda n: degree_dict.get(n, 0), reverse=True)
            
            for n in neighbors:
                if n not in selected_nodes:
                    selected_nodes.add(n)
                    queue.append(n)
                    if len(selected_nodes) >= self.max_size:
                        break
        
        return list(selected_nodes)

    def _extract_subgraph_from_nodes(self, nodes):
        """
        Extract subgraph objects, adjacency matrix, and mapping from the node list.
        """
        subgraph = self.G.subgraph(nodes).copy()
        
        # Reorder nodes to ensure matrix index alignment
        sorted_nodes = sorted(list(subgraph.nodes()))
        
        # Extracting the adjacency matrix (numpy array)
        adj_matrix = nx.to_numpy_array(subgraph, nodelist=sorted_nodes)
        
        # Establish local index mapping: Subgraph Index -> Protein Name
        idx_map = {i: name for i, name in enumerate(sorted_nodes)}
        
        # Returns: A NetworkX object, an adjacency matrix, a list of nodes, and an index map.
        return subgraph, adj_matrix, sorted_nodes, idx_map
