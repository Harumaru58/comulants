"""
Manual Implementation of Alternating Least Squares (ALS) for CP Decomposition

This script implements CP tensor decomposition using the ALS algorithm without
relying on tensorly's built-in functions.
"""

import numpy as np
from scipy.linalg import pinv


def khatri_rao_product(matrices):
    """
    Compute the Khatri-Rao product of multiple matrices.
    
    For matrices A (I×R) and B (J×R), the Khatri-Rao product A ⊙ B is (IJ×R).
    
    Parameters:
    -----------
    matrices : list of arrays
        List of matrices, each of shape (n_i, rank)
        
    Returns:
    --------
    result : array
        Khatri-Rao product of shape (∏n_i, rank)
    """
    if len(matrices) == 0:
        raise ValueError("Need at least one matrix")
    if len(matrices) == 1:
        return matrices[0]
    
    result = matrices[0]
    for mat in matrices[1:]:
        I, R = result.shape
        J, R2 = mat.shape
        if R != R2:
            raise ValueError("All matrices must have same number of columns")
        
        # Khatri-Rao: (A ⊙ B)[i*J + j, r] = A[i, r] * B[j, r]
        result = np.einsum('ir,jr->ijr', result, mat).reshape(I * J, R)
    
    return result


def unfold_tensor(tensor, mode):
    """
    Unfold a tensor along a given mode.
    
    Parameters:
    -----------
    tensor : array
        N-way tensor
    mode : int
        Mode along which to unfold (0-indexed)
        
    Returns:
    --------
    unfolded : array
        Matrix of shape (n_mode, ∏n_i for i≠mode)
    """
    tensor = np.asarray(tensor)
    n_modes = tensor.ndim
    
    if mode < 0 or mode >= n_modes:
        raise ValueError(f"Mode must be between 0 and {n_modes-1}")
    
    # Permute so mode becomes first dimension
    perm = [mode] + [i for i in range(n_modes) if i != mode]
    tensor_permuted = np.transpose(tensor, perm)
    
    # Reshape to (n_mode, -1)
    shape = tensor_permuted.shape
    unfolded = tensor_permuted.reshape(shape[0], -1)
    
    return unfolded


def fold_tensor(unfolded, mode, shape):
    """
    Fold an unfolded tensor back to original shape.
    
    Parameters:
    -----------
    unfolded : array
        Unfolded matrix
    mode : int
        Mode along which it was unfolded
    shape : tuple
        Original tensor shape
        
    Returns:
    --------
    tensor : array
        Folded tensor
    """
    n_modes = len(shape)
    perm = [mode] + [i for i in range(n_modes) if i != mode]
    inverse_perm = np.argsort(perm)
    
    # Reshape to permuted shape
    permuted_shape = [shape[i] for i in perm]
    tensor_permuted = unfolded.reshape(permuted_shape)
    
    # Permute back
    tensor = np.transpose(tensor_permuted, inverse_perm)
    
    return tensor


def initialize_factors(tensor_shape, rank, init='random', random_state=None):
    """
    Initialize factor matrices for CP decomposition.
    
    Parameters:
    -----------
    tensor_shape : tuple
        Shape of the tensor
    rank : int
        Rank of decomposition
    init : str
        Initialization method: 'random' or 'svd'
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    factors : list of arrays
        List of factor matrices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    factors = []
    
    if init == 'svd':
        # Use SVD-based initialization for first mode
        tensor_0 = unfold_tensor(np.random.randn(*tensor_shape), 0)
        U, S, Vt = np.linalg.svd(tensor_0, full_matrices=False)
        factors.append(U[:, :rank])
        
        # Initialize others randomly
        for i in range(1, len(tensor_shape)):
            factors.append(np.random.randn(tensor_shape[i], rank))
    else:
        # Random initialization
        for n in tensor_shape:
            factors.append(np.random.randn(n, rank))
    
    return factors


def cp_reconstruct(factors, weights=None):
    """
    Reconstruct tensor from CP factors.
    
    Parameters:
    -----------
    factors : list of arrays
        Factor matrices
    weights : array, optional
        Component weights. If None, uses ones.
        
    Returns:
    --------
    tensor : array
        Reconstructed tensor
    """
    rank = factors[0].shape[1]
    
    if weights is None:
        weights = np.ones(rank)
    
    tensor = np.zeros([f.shape[0] for f in factors])
    
    for r in range(rank):
        # Outer product of r-th column from each factor
        component = factors[0][:, r]
        for f in factors[1:]:
            component = np.outer(component, f[:, r]).flatten()
        component = component.reshape([f.shape[0] for f in factors])
        tensor += weights[r] * component
    
    return tensor


def als_cp_decomposition(tensor, rank, max_iter=1000, tol=1e-6, 
                         init='random', random_state=42, verbose=False):
    """
    CP decomposition using Alternating Least Squares (ALS).
    
    For a tensor T of shape (I, J, K) and rank R:
    T ≈ Σᵣ λᵣ aᵣ ⊗ bᵣ ⊗ cᵣ
    
    where:
    - aᵣ ∈ ℝᴵ, bᵣ ∈ ℝᴶ, cᵣ ∈ ℝᴷ are factor vectors
    - λᵣ are component weights
    
    Parameters:
    -----------
    tensor : array
        N-way tensor to decompose
    rank : int
        Rank of decomposition
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance (relative change in error)
    init : str
        Initialization method: 'random' or 'svd'
    random_state : int
        Random seed for initialization
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'factors': list of factor matrices
        - 'weights': component weights
        - 'reconstruction': reconstructed tensor
        - 'relative_error': relative reconstruction error
        - 'n_iter': number of iterations
        - 'converged': whether algorithm converged
    """
    tensor = np.asarray(tensor)
    tensor_shape = tensor.shape
    n_modes = len(tensor_shape)
    tensor_norm = np.linalg.norm(tensor)
    
    if tensor_norm < 1e-10:
        raise ValueError("Tensor norm too small")
    
    # Initialize factors
    factors = initialize_factors(tensor_shape, rank, init=init, random_state=random_state)
    
    # Normalize factors
    for i in range(n_modes):
        for r in range(rank):
            norm = np.linalg.norm(factors[i][:, r])
            if norm > 1e-10:
                factors[i][:, r] /= norm
    
    prev_error = np.inf
    converged = False
    
    for iteration in range(max_iter):
        # Update each factor matrix in turn
        for mode in range(n_modes):
            # Unfold tensor along this mode
            tensor_unfolded = unfold_tensor(tensor, mode)
            
            # Compute Khatri-Rao product of all other factors
            other_factors = [factors[i] for i in range(n_modes) if i != mode]
            khatri_rao = khatri_rao_product(other_factors)
            
            # Solve: tensor_unfolded ≈ factors[mode] @ khatri_rao.T
            # Using pseudo-inverse: factors[mode] = tensor_unfolded @ pinv(khatri_rao.T)
            # Which is: factors[mode] = tensor_unfolded @ khatri_rao @ pinv(khatri_rao.T @ khatri_rao)
            
            # More numerically stable: solve normal equations
            gram = khatri_rao.T @ khatri_rao
            rhs = tensor_unfolded @ khatri_rao
            
            try:
                factors[mode] = rhs @ pinv(gram)
            except:
                # Fallback to direct pseudo-inverse
                factors[mode] = tensor_unfolded @ pinv(khatri_rao.T)
            
            # Normalize columns and extract weights
            weights = np.zeros(rank)
            for r in range(rank):
                norm = np.linalg.norm(factors[mode][:, r])
                if norm > 1e-10:
                    weights[r] = norm
                    factors[mode][:, r] /= norm
                else:
                    weights[r] = 0.0
        
        # Compute reconstruction error
        reconstruction = cp_reconstruct(factors, weights)
        error = np.linalg.norm(tensor - reconstruction)
        relative_error = error / tensor_norm
        
        # Check convergence
        with np.errstate(divide='ignore', invalid='ignore'):
            if prev_error > 1e-10 and not np.isinf(prev_error):
                error_change = abs(prev_error - relative_error) / prev_error
            else:
                error_change = abs(relative_error) if not np.isinf(prev_error) else 1.0
        if error_change < tol:
            converged = True
            if verbose:
                print(f"Converged at iteration {iteration+1}")
            break
        
        prev_error = relative_error
        
        if verbose and (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration+1}: relative error = {relative_error:.6f}")
    
    return {
        'factors': factors,
        'weights': weights,
        'reconstruction': reconstruction,
        'relative_error': relative_error,
        'n_iter': iteration + 1,
        'converged': converged
    }


def main():
    """Example usage of ALS CP decomposition."""
    # Create a simple test tensor
    np.random.seed(42)
    
    # Create a rank-3 tensor: sum of 3 rank-1 tensors
    I, J, K = 10, 15, 20
    rank = 3
    
    factors_true = [
        np.random.randn(I, rank),
        np.random.randn(J, rank),
        np.random.randn(K, rank)
    ]
    weights_true = np.array([2.0, 1.5, 1.0])
    
    # Create tensor
    tensor = cp_reconstruct(factors_true, weights_true)
    
    # Add noise
    noise_level = 0.1
    tensor += noise_level * np.random.randn(*tensor.shape) * np.std(tensor)
    
    print("Testing ALS CP Decomposition")
    print("=" * 50)
    print(f"Tensor shape: {tensor.shape}")
    print(f"True rank: {rank}")
    print(f"Noise level: {noise_level}")
    print()
    
    # Decompose
    result = als_cp_decomposition(tensor, rank=rank, max_iter=1000, tol=1e-6, 
                                  init='svd', random_state=42, verbose=True)
    
    print()
    print("Results:")
    print(f"  Relative error: {result['relative_error']:.6f}")
    print(f"  Iterations: {result['n_iter']}")
    print(f"  Converged: {result['converged']}")
    print(f"  Component weights: {result['weights']}")
    print()
    print("Factor matrix shapes:")
    for i, factor in enumerate(result['factors']):
        print(f"  Factor {i}: {factor.shape}")


if __name__ == "__main__":
    main()

