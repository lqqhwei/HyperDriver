# resource/energy_case_study/src/methods.py
import random
import numpy as np
import scipy.linalg as la

class DriverSelector:
    def __init__(self, G_sub, A_sub, node_list):
        """
        G_sub: NetworkX subgraph object
        A_sub: Subgraph adjacency matrix (NumPy array)
        node_list: List of node names (corresponding to A_sub indices)
        """
        self.G = G_sub
        self.A = A_sub
        self.nodes = node_list
        self.num_nodes = len(node_list)

    def select_random(self, k, seed=42):
        """[Benchmark 1] Random Selection"""
        random.seed(seed)
        indices = list(range(self.num_nodes))
        selected_indices = random.sample(indices, k)
        return selected_indices, [self.nodes[i] for i in selected_indices]

    def select_degree(self, k):
        """[Benchmark 2] Degree Centrality (Hubs)"""
        degrees = np.sum(self.A, axis=1)
        # Sort desc
        sorted_indices = np.argsort(degrees)[::-1]
        selected_indices = sorted_indices[:k]
        return selected_indices, [self.nodes[i] for i in selected_indices]

    def select_hyperdriver(self, k):
        """
        [Proposed Method] HyperDriver Global Greedy (V17 Logic)
        Iteratively selects nodes that minimize the Control Energy (Tr(W_c^-1)).
        Strictly aligned with control_engine.py V17.
        """
        # 1. Construct System Matrix (Laplacian Dynamics)
        # L = D - A
        degrees = np.sum(self.A, axis=1)
        L = np.diag(degrees) - self.A
        
        # 2. Spectral Decomposition
        # We use a small epsilon for stability, similar to control_engine.py
        epsilon = 1e-9
        L_sys = L + epsilon * np.eye(self.num_nodes)
        
        # Eigen-decomposition
        eigvals, eigvecs = la.eigh(L_sys)
        
        # Filter small eigenvalues to avoid numerical instability
        valid_mask = eigvals > 1e-9
        lambdas = eigvals[valid_mask]
        V_sq = np.square(eigvecs[:, valid_mask])
        
        # 3. Global Greedy Iteration
        # Objective: Minimize Energy = Sum( 1 / (Sum(v_i^2) + epsilon) ) * weighting
        # V17 Formula: Energy ~ Sum( 2*lambda / (Current_Proj + Candidate_Proj) )
        
        two_lambdas = 2.0 * lambdas
        current_proj = np.zeros(len(lambdas))
        remaining_indices = list(range(self.num_nodes))
        selected_indices = []

        for _ in range(k):
            # Calculate energy for all remaining candidates
            # Shape: [Num_Remaining]
            # We want to find the node that results in the MINIMUM total energy
            candidate_proj = V_sq[remaining_indices] # [M, Num_Eig]
            
            # Broadcast addition: (1, Num_Eig) + (M, Num_Eig)
            total_proj = current_proj + candidate_proj + 1e-12
            
            # Energy = Sum( 2*lambda / Projection )
            energies = np.sum(two_lambdas / total_proj, axis=1)
            
            # Greedy Choice: Argmin Energy
            best_local_idx = np.argmin(energies)
            best_global_idx = remaining_indices.pop(best_local_idx)
            
            # Update state
            selected_indices.append(best_global_idx)
            current_proj += V_sq[best_global_idx]
            
        return selected_indices, [self.nodes[i] for i in selected_indices]