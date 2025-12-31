import random
import numpy as np
import networkx as nx

class DriverSelector:
    def __init__(self, G_sub, A_sub, node_list):
        """
        G_sub: NetworkX subgraph objects
        A_sub: Subgraph adjacency matrix (NumPy array)
        node_list: List of node names (corresponding to the A_sub index)
        """
        self.G = G_sub
        self.A = A_sub
        self.nodes = node_list
        self.num_nodes = len(node_list)

    def select_random(self, k, seed=42):
        """
        [Benchmark 1] Randomly select K nodes
        """
        random.seed(seed)
        indices = list(range(self.num_nodes))
        selected_indices = random.sample(indices, k)
        return selected_indices, [self.nodes[i] for i in selected_indices]

    def select_degree(self, k):
        """
        [Benchmark 2] Select K nodes based on degree centrality (Hubs)
        """
        # Calculate the degree (using NetworkX or directly sum(A)).
        degrees = np.sum(self.A, axis=1)
        
        # Retrieve index sorting (from largest to smallest)
        # argsort sorts by default from smallest to largest, so inverting or slicing [::-1] is an option.
        sorted_indices = np.argsort(degrees)[::-1]
        
        selected_indices = sorted_indices[:k]
        return selected_indices, [self.nodes[i] for i in selected_indices]

    def select_hyperdriver_proxy(self, k):
        """
        [HyperDriver Logic] Selection based on spectral energy proxy
        
        Thesis basis: 
        1. Energy Proxy E ~ 1 / lambda_max (Section 2.4.2) [cite: 91]
        2. Driver rating K_i ~ Delta E_i (Section 2.4.3) [cite: 100]
        
        logic:
        We need to find those nodes that, if removed, would cause the network energy E to deteriorate the most.
        This means that these nodes are the key anchors for maintaining the current "low-energy controllable state" of the network.
        Therefore, controlling them (Input) is the most effective approach.
        """
        print("Calculating Spectral Energy Proxy scores...")
        
        # 1. Calculate the original Laplacian matrix and its largest eigenvalue.
        # L = D - A
        D = np.diag(np.sum(self.A, axis=1))
        L_base = D - self.A
        
        # Calculate the original spectral radius (Largest Eigenvalue of Laplacian)
        # Note: For an undirected graph L, which is positive semi-definite, all eigenvalues ​​are real numbers >= 0.
        try:
            # Using eigvalsh to compute eigenvalues ​​of symmetric matrices is faster and more stable.
            evals_base = np.linalg.eigvalsh(L_base)
            lambda_max_base = np.max(evals_base)
        except np.linalg.LinAlgError:
            lambda_max_base = 0.0

        scores = []
        
        # 2. Iterate through each node and calculate the energy change "after the perturbation".
        for i in range(self.num_nodes):
            # Simulate removing node i:
            # Delete the i-th row and i-th column from the matrix.
            # In fact, we construct an (N-1)x(N-1) matrix
            A_prime = np.delete(np.delete(self.A, i, axis=0), i, axis=1)
            D_prime_vals = np.sum(A_prime, axis=1)
            L_prime = np.diag(D_prime_vals) - A_prime
            
            try:
                evals_prime = np.linalg.eigvalsh(L_prime)
                lambda_max_prime = np.max(evals_prime)
            except:
                lambda_max_prime = 0.0
            
            # Calculate the score:
            # Based on the paper, we focus on Delta E.
            # E_base ~ 1/lambda_max_base
            # E_prime ~ 1/lambda_max_prime
            # If a node is important, removing it will make the system "more difficult to control" (Structure degrades), 
            # This typically manifests as a decrease in lambda_max (Connectivity/Stiffness drops).
            # This causes E_prime (1/small) to become very large.
            # Score = E_prime - E_base
            
            epsilon = 1e-9
            energy_base = 1.0 / (lambda_max_base + epsilon)
            energy_prime = 1.0 / (lambda_max_prime + epsilon)
            
            delta_E = energy_prime - energy_base
            scores.append((i, delta_E))
            
        # 3. Sort by selecting the node with the largest Delta E (i.e., the node with the highest removal cost).
        scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_indices = [idx for idx, score in scores[:k]]
        return selected_indices, [self.nodes[i] for i in selected_indices]

# Test code
if __name__ == "__main__":
    # Simple Mock Data Test
    G_mock = nx.erdos_renyi_graph(20, 0.3, seed=42)
    A_mock = nx.to_numpy_array(G_mock)
    nodes_mock = [str(i) for i in range(20)]
    
    selector = DriverSelector(G_mock, A_mock, nodes_mock)
    
    k = 3
    print("Random:", selector.select_random(k)[0])
    print("Degree:", selector.select_degree(k)[0])
    print("HyperDriver:", selector.select_hyperdriver_proxy(k)[0])