import numpy as np
from scipy import linalg

class ControlEnergyCalculator:
    def __init__(self, adj_matrix):
        """
        Initialize the energy calculator
        adj_matrix: The adjacency matrix of the subgraph (NxN NumPy array)
        """
        self.A_adj = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        
        # 1. Construct the system matrix A_sys
        # Kinetic equation: dx/dt = -L x (Consensus protocol/diffusion process)
        # L = D - A
        degrees = np.sum(adj_matrix, axis=1)
        self.L = np.diag(degrees) - adj_matrix
        
        # 2. Stability Perturbation
        # To ensure that the Lyapunov equation has a unique solution, matrix A must be Hurwitz stable (real eigenvalues ​​< 0).
        # The eigenvalues ​​of the original -L are <= 0 and include 0.
        # We introduce a small decay rate, epsilon (simulating biodegradation), to gradually stabilize the system.
        self.epsilon = 0.5  # For PPI networks, a decay factor of 0.1 to 1.0 is reasonable.
        self.A_sys = -self.L - self.epsilon * np.eye(self.num_nodes)

    def compute_energy(self, driver_indices):
        """
        Calculate the "average control energy" for a given set of driving nodes.
        driver_indices: list of int, the index of the selected driver node.
        
        Returns:
            energy_score: float (smaller values ​​are better)
        """
        k = len(driver_indices)
        if k == 0:
            return float('inf')

        # 1. Construct the input matrix B (NxK)
        # Only the row corresponding to driver_indices is 1, the rest are 0.
        B = np.zeros((self.num_nodes, k))
        for col_idx, node_idx in enumerate(driver_indices):
            B[node_idx, col_idx] = 1.0

        # 2. Solve the continuous Lyapunov equation.
        # A X + X A^T = Q
        # Corresponding to our format: A_sys Wc + Wc A_sys^T + B B^T = 0
        # Therefore, Q = -B B^T
        Q = -np.dot(B, B.T)
        
        try:
            # scipy.linalg.solve_continuous_lyapunov(a, q) solves AX + XA^H = Q
            Wc = linalg.solve_continuous_lyapunov(self.A_sys, Q)
        except Exception as e:
            print(f"Lyapunov Solver failed: {e}")
            return float('inf')

        # 3. Calculate energy index
        # Theory: Minimum Energy ~ Trace(Inv(Wc))
        # The eigenvalues ​​of Gramian Wc measure the controllability of a system in all directions.
        # Larger eigenvalues ​​mean better controllability and less energy required.
        # The energy E is inversely proportional to the eigenvalue of Wc.
        
        try:
            # Calculate the eigenvalues ​​of Wc
            evals = linalg.eigvalsh(Wc)
            
            # Filter out extremely small eigenvalues ​​to avoid division by zero (numerical truncation).
            # In actual calculations, Wc should be positive definite, but floating-point errors may produce tiny negative numbers or zeros.
            min_tol = 1e-12
            evals = evals[evals > min_tol]
            
            if len(evals) == 0:
                return float('inf')
            
            # Metrics: Average Energy = Trace(Wc^-1) = Sum(1/lambda_i)
            energy = np.sum(1.0 / evals)
            
            return energy

        except np.linalg.LinAlgError:
            return float('inf')

# Test code
if __name__ == "__main__":
    # Create a small test graph
    import networkx as nx
    G_test = nx.path_graph(10)
    A_test = nx.to_numpy_array(G_test)
    
    calculator = ControlEnergyCalculator(A_test)
    
    # Comparison of two driving schemes
    # Option A: Select endpoints (usually more difficult to control the entire map)
    drivers_A = [0] 
    energy_A = calculator.compute_energy(drivers_A)
    
    # Option B: Select a center point (usually easier to control)
    drivers_B = [4]
    energy_B = calculator.compute_energy(drivers_B)
    
    print(f"Energy (End Node): {energy_A:.4e}")
    print(f"Energy (Center Node): {energy_B:.4e}")
    
    if energy_B < energy_A:
        print("Test passed: The central node controls lower energy consumption.")
    else:
        print("Test failed: The result is not intuitive.")