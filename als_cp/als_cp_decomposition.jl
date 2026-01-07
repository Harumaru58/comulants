"""
Manual Implementation of Alternating Least Squares (ALS) for CP Decomposition

This script implements CP tensor decomposition using the ALS algorithm 
"""

using LinearAlgebra
using Random
using Printf
using Statistics

"""
    khatri_rao_product(matrices)

Compute the Khatri-Rao product of multiple matrices.

For matrices A (I×R) and B (J×R), the Khatri-Rao product A ⊙ B is (IJ×R).

# Arguments
- `matrices`: Vector of matrices, each of shape (n_i, rank)

# Returns
- `result`: Khatri-Rao product of shape (∏n_i, rank)
"""
function khatri_rao_product(matrices::Vector{Matrix{Float64}})
    if isempty(matrices)
        error("Need at least one matrix")
    end
    if length(matrices) == 1
        return matrices[1]
    end
    
    result = matrices[1]
    for mat in matrices[2:end]
        I, R = size(result)
        J, R2 = size(mat)
        if R != R2
            error("All matrices must have same number of columns")
        end
        
        # Khatri-Rao: (A ⊙ B)[i*J + j, r] = A[i, r] * B[j, r]
        # Using broadcasting and reshape
        result = reshape(
            [result[i, r] * mat[j, r] for i in 1:I, j in 1:J, r in 1:R],
            I * J, R
        )
    end
    
    return result
end

"""
    unfold_tensor(tensor, mode)

Unfold a tensor along a given mode.

# Arguments
- `tensor`: N-way tensor (Array)
- `mode`: Mode along which to unfold (1-indexed in Julia)

# Returns
- `unfolded`: Matrix of shape (n_mode, ∏n_i for i≠mode)
"""
function unfold_tensor(tensor::Array, mode::Int)
    n_modes = ndims(tensor)
    
    if mode < 1 || mode > n_modes
        error("Mode must be between 1 and $n_modes")
    end
    
    # Permute so mode becomes first dimension
    perm = [mode; [i for i in 1:n_modes if i != mode]]
    tensor_permuted = permutedims(tensor, perm)
    
    # Reshape to (n_mode, -1)
    shape = size(tensor_permuted)
    unfolded = reshape(tensor_permuted, shape[1], :)
    
    return unfolded
end

"""
    fold_tensor(unfolded, mode, shape)

Fold an unfolded tensor back to original shape.

# Arguments
- `unfolded`: Unfolded matrix
- `mode`: Mode along which it was unfolded
- `shape`: Original tensor shape (tuple)

# Returns
- `tensor`: Folded tensor
"""
function fold_tensor(unfolded::Matrix, mode::Int, shape::Tuple)
    n_modes = length(shape)
    perm = [mode; [i for i in 1:n_modes if i != mode]]
    inverse_perm = sortperm(perm)
    
    # Reshape to permuted shape
    permuted_shape = [shape[i] for i in perm]
    tensor_permuted = reshape(unfolded, permuted_shape...)
    
    # Permute back
    tensor = permutedims(tensor_permuted, inverse_perm)
    
    return tensor
end

"""
    initialize_factors(tensor_shape, rank; init=:random, rng=MersenneTwister(42))

Initialize factor matrices for CP decomposition.

# Arguments
- `tensor_shape`: Shape of the tensor (tuple)
- `rank`: Rank of decomposition
- `init`: Initialization method (`:random` or `:svd`)
- `rng`: Random number generator

# Returns
- `factors`: Vector of factor matrices
"""
function initialize_factors(tensor_shape::Tuple, rank::Int; 
                           init::Symbol=:random, rng::AbstractRNG=MersenneTwister(42))
    factors = Matrix{Float64}[]
    
    if init == :svd
        # Use SVD-based initialization for first mode
        temp_tensor = randn(rng, tensor_shape...)
        tensor_0 = unfold_tensor(temp_tensor, 1)
        U, S, Vt = svd(tensor_0)
        push!(factors, U[:, 1:rank])
        
        # Initialize others randomly
        for i in 2:length(tensor_shape)
            push!(factors, randn(rng, tensor_shape[i], rank))
        end
    else
        # Random initialization
        for n in tensor_shape
            push!(factors, randn(rng, n, rank))
        end
    end
    
    # Normalize factors
    for i in 1:length(factors)
        for r in 1:rank
            norm_val = norm(factors[i][:, r])
            if norm_val > 1e-10
                factors[i][:, r] ./= norm_val
            end
        end
    end
    
    return factors
end

"""
    cp_reconstruct(factors, weights=nothing)

Reconstruct tensor from CP factors.

# Arguments
- `factors`: Vector of factor matrices
- `weights`: Component weights (optional). If nothing, uses ones.

# Returns
- `tensor`: Reconstructed tensor
"""
function cp_reconstruct(factors::Vector{Matrix{Float64}}, weights::Union{Vector{Float64}, Nothing}=nothing)
    rank = size(factors[1], 2)
    tensor_shape = [size(f, 1) for f in factors]
    
    if weights === nothing
        weights = ones(rank)
    end
    
    # Initialize tensor
    tensor = zeros(tensor_shape...)
    
    for r in 1:rank
        # Outer product of r-th column from each factor
        component = factors[1][:, r]
        for f in factors[2:end]
            component = vec(component * f[:, r]')
        end
        component = reshape(component, tensor_shape...)
        tensor .+= weights[r] .* component
    end
    
    return tensor
end

"""
    als_cp_decomposition(tensor, rank; max_iter=1000, tol=1e-6, 
                        init=:random, rng=MersenneTwister(42), verbose=false)

CP decomposition using Alternating Least Squares (ALS).

For a tensor T of shape (I, J, K) and rank R:
T ≈ Σᵣ λᵣ aᵣ ⊗ bᵣ ⊗ cᵣ

where:
- aᵣ ∈ ℝᴵ, bᵣ ∈ ℝᴶ, cᵣ ∈ ℝᴷ are factor vectors
- λᵣ are component weights

# Arguments
- `tensor`: N-way tensor to decompose
- `rank`: Rank of decomposition
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance (relative change in error)
- `init`: Initialization method (`:random` or `:svd`)
- `rng`: Random number generator
- `verbose`: Whether to print progress

# Returns
- Dictionary containing:
  - `factors`: Vector of factor matrices
  - `weights`: Component weights
  - `reconstruction`: Reconstructed tensor
  - `relative_error`: Relative reconstruction error
  - `n_iter`: Number of iterations
  - `converged`: Whether algorithm converged
"""
function als_cp_decomposition(tensor::Array, rank::Int; 
                             max_iter::Int=1000, tol::Float64=1e-6,
                             init::Symbol=:random, rng::AbstractRNG=MersenneTwister(42),
                             verbose::Bool=false)
    tensor_shape = size(tensor)
    n_modes = length(tensor_shape)
    tensor_norm = norm(tensor)
    
    if tensor_norm < 1e-10
        error("Tensor norm too small")
    end
    
    # Initialize factors
    factors = initialize_factors(tensor_shape, rank; init=init, rng=rng)
    
    prev_error = Inf
    converged = false
    relative_error = 0.0
    reconstruction = similar(tensor)
    weights = ones(rank)  # Initialize weights
    final_iteration = max_iter  # Default to max_iter if not converged
    
    for iteration in 1:max_iter
        # Update each factor matrix in turn
        for mode in 1:n_modes
            # Unfold tensor along this mode
            tensor_unfolded = unfold_tensor(tensor, mode)
            
            # Compute Khatri-Rao product of all other factors
            other_factors = [factors[i] for i in 1:n_modes if i != mode]
            khatri_rao = khatri_rao_product(other_factors)
            
            # Solve: tensor_unfolded ≈ factors[mode] * khatri_rao'
            # Using pseudo-inverse: factors[mode] = tensor_unfolded * pinv(khatri_rao')
            # More numerically stable: solve normal equations
            gram = khatri_rao' * khatri_rao
            rhs = tensor_unfolded * khatri_rao
            
            try
                factors[mode] = rhs * pinv(gram)
            catch
                # Fallback to direct pseudo-inverse
                factors[mode] = tensor_unfolded * pinv(khatri_rao')
            end
            
            # Normalize columns and extract weights
            for r in 1:rank
                norm_val = norm(factors[mode][:, r])
                if norm_val > 1e-10
                    weights[r] = norm_val
                    factors[mode][:, r] ./= norm_val
                else
                    weights[r] = 0.0
                end
            end
        end
        
        # Compute reconstruction error
        reconstruction = cp_reconstruct(factors, weights)
        error_val = norm(tensor - reconstruction)
        relative_error = error_val / tensor_norm
        
        # Check convergence
        if prev_error > 1e-10 && !isinf(prev_error)
            error_change = abs(prev_error - relative_error) / prev_error
        else
            error_change = isinf(prev_error) ? 1.0 : abs(relative_error)
        end
        
        if error_change < tol
            converged = true
            final_iteration = iteration
            if verbose
                println("Converged at iteration $iteration")
            end
            break
        end
        
        prev_error = relative_error
        
        if verbose && iteration % 100 == 0
            @printf("Iteration %d: relative error = %.6f\n", iteration, relative_error)
        end
    end
    
    return Dict(
        :factors => factors,
        :weights => weights,
        :reconstruction => reconstruction,
        :relative_error => relative_error,
        :n_iter => final_iteration,
        :converged => converged
    )
end

# Example usage
function main()
    println("Testing ALS CP Decomposition")
    println("=" ^ 50)
    
    # Create a simple test tensor
    Random.seed!(42)
    
    # Create a rank-3 tensor: sum of 3 rank-1 tensors
    I, J, K = 10, 15, 20
    rank = 3
    
    factors_true = [
        randn(I, rank),
        randn(J, rank),
        randn(K, rank)
    ]
    weights_true = [2.0, 1.5, 1.0]
    
    # Create tensor
    tensor = cp_reconstruct(factors_true, weights_true)
    
    # Add noise
    noise_level = 0.1
    tensor .+= noise_level .* randn(size(tensor)...) .* std(tensor)
    
    println("Tensor shape: $(size(tensor))")
    println("True rank: $rank")
    println("Noise level: $noise_level")
    println()
    
    # Decompose
    result = als_cp_decomposition(tensor, rank; max_iter=1000, tol=1e-6,
                                  init=:svd, rng=MersenneTwister(42), verbose=true)
    
    println()
    println("Results:")
    @printf("  Relative error: %.6f\n", result[:relative_error])
    println("  Iterations: $(result[:n_iter])")
    println("  Converged: $(result[:converged])")
    println("  Component weights: $(result[:weights])")
    println()
    println("Factor matrix shapes:")
    for (i, factor) in enumerate(result[:factors])
        println("  Factor $i: $(size(factor))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

