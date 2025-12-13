"""
Symmetric CP decomposition using ALS - all factor matrices constrained to be equal

For a symmetric tensor T of shape (d, d, ..., d) and rank R:
T ≈ Σᵣ λᵣ aᵣ ⊗ aᵣ ⊗ ... ⊗ aᵣ

where all factor matrices are the same: A = [a₁, a₂, ..., aᵣ]
"""

using LinearAlgebra
using Random
using Printf

# Include helper functions from als_cp_dec.jl
include("als_cp_dec.jl")

"""
    symmetric_als_cp(tensor, rank; max_iter=1000, tol=1e-6, verbose=false)

Symmetric CP decomposition where all factor matrices are constrained to be equal.

# Arguments
- `tensor`: N-way symmetric tensor (all dimensions must be equal)
- `rank`: Rank of decomposition
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance (absolute change in error)
- `verbose`: Whether to print progress

# Returns
- Named tuple containing:
  - `factor`: Single factor matrix (d × rank)
  - `reconstruction`: Reconstructed tensor
  - `error`: Relative reconstruction error
  - `n_iter`: Number of iterations
  - `converged`: Whether algorithm converged
"""
function symmetric_als_cp(tensor::Array, rank::Int; 
                         max_iter::Int=1000, tol::Float64=1e-6, 
                         verbose::Bool=false)
    n_modes = ndims(tensor)
    d = size(tensor, 1)
    
    # All modes must have same dimension for symmetric tensor
    @assert all(size(tensor, i) == d for i in 1:n_modes) "Not a symmetric tensor shape"
    
    # Initialize single factor matrix
    A = randn(d, rank)
    for r in 1:rank
        A[:, r] ./= norm(A[:, r])
    end
    
    tensor_norm = norm(tensor)
    prev_error = Inf
    weights = ones(rank)
    
    for iter in 1:max_iter
        # Update factor matrix by averaging across all modes
        A_new = zeros(d, rank)
        
        for mode in 1:n_modes
            X_unfold = unfold_tensor(tensor, mode)
            # All other factors are the same A
            other_factors = [A for _ in 1:(n_modes-1)]
            Z = khatri_rao_product(other_factors)
            
            # Solve least squares: X_unfold ≈ A_mode * Z'
            # More numerically stable: solve normal equations
            gram = Z' * Z
            rhs = X_unfold * Z
            
            A_mode = try
                rhs * pinv(gram)
            catch
                # Fallback to direct pseudo-inverse
                X_unfold * pinv(Z')
            end
            
            A_new .+= A_mode
        end
        
        # Average across modes
        A = A_new / n_modes
        
        # Normalize columns and extract weights
        for r in 1:rank
            norm_val = norm(A[:, r])
            if norm_val > 1e-10
                weights[r] = norm_val
                A[:, r] ./= norm_val
            else
                weights[r] = 0.0
            end
        end
        
        # Check convergence
        factors_all = fill(A, n_modes)
        reconstruction = cp_reconstruct(factors_all, weights)
        error = norm(tensor - reconstruction) / tensor_norm
        
        # Relative change in error
        if prev_error > 1e-10 && !isinf(prev_error)
            error_change = abs(prev_error - error) / prev_error
        else
            error_change = isinf(prev_error) ? 1.0 : abs(error)
        end
        
        if error_change < tol
            verbose && println("Converged at iteration $iter")
            return (factor=A, weights=weights, reconstruction=reconstruction, 
                   error=error, n_iter=iter, converged=true)
        end
        
        prev_error = error
        verbose && iter % 100 == 0 && 
            @printf("Iter %d: error = %.6f\n", iter, error)
    end
    
    return (factor=A, weights=weights, reconstruction=cp_reconstruct(fill(A, n_modes), weights), 
           error=prev_error, n_iter=max_iter, converged=false)
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("Testing Symmetric ALS CP Decomposition")
    println("=" ^ 50)
    
    Random.seed!(42)
    
    # Create a symmetric 3-way tensor
    d = 10
    rank = 3
    
    # Create symmetric tensor from single factor matrix
    A_true = randn(d, rank)
    for r in 1:rank
        A_true[:, r] ./= norm(A_true[:, r])
    end
    
    # Create symmetric tensor: T = Σᵣ aᵣ ⊗ aᵣ ⊗ aᵣ
    tensor = zeros(d, d, d)
    for r in 1:rank
        component = A_true[:, r]
        component = kron(component, component)
        component = kron(component, A_true[:, r])
        tensor .+= reshape(component, d, d, d)
    end
    
    # Add noise
    noise_level = 0.1
    tensor .+= noise_level .* randn(size(tensor)...) .* std(tensor)
    
    println("Tensor shape: $(size(tensor))")
    println("Rank: $rank")
    println("Noise level: $noise_level")
    println()
    
    # Decompose
    result = symmetric_als_cp(tensor, rank; max_iter=1000, tol=1e-6, verbose=true)
    
    println()
    println("Results:")
    @printf("  Relative error: %.6f\n", result.error)
    println("  Iterations: $(result.n_iter)")
    println("  Converged: $(result.converged)")
    println("  Factor matrix shape: $(size(result.factor))")
end

