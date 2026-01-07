"""
Manual Implementation of Alternating Least Squares (ALS) for CP Decomposition
"""

using LinearAlgebra
using Random
using Printf
using Statistics

"""
    khatri_rao_product(matrices)
"""
function khatri_rao_product(matrices::Vector{Matrix{Float64}})
    isempty(matrices) && error("Need at least one matrix")
    
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
Unfold a tensor along a given mode.

"""
function unfold_tensor(tensor::Array, mode::Int)
    n_modes = ndims(tensor)
    (mode < 1 || mode > n_modes) && error("Mode must be between 1 and $n_modes")
    
    perm = [mode; setdiff(1:n_modes, mode)]
    return reshape(permutedims(tensor, perm), size(tensor, mode), :)
end

"""
Fold an unfolded tensor back to original shape.

"""
function fold_tensor(unfolded::Matrix, mode::Int, shape::Tuple)
    perm = [mode; setdiff(1:length(shape), mode)]
    tensor_permuted = reshape(unfolded, [shape[i] for i in perm]...)
    return permutedims(tensor_permuted, sortperm(perm))
end

"""
Initialize factor matrices for CP decomposition.

"""
function initialize_factors(tensor_shape::Tuple, rank::Int; 
                           init::Symbol=:random, rng::AbstractRNG=MersenneTwister(42))
    factors = [randn(rng, n, rank) for n in tensor_shape]
    for A in factors, r in 1:rank
        (norm_val = norm(A[:, r])) > 1e-10 && (A[:, r] ./= norm_val)
    end
    return factors
end

"""

Reconstruct tensor from CP factors.
"""
function cp_reconstruct(factors::Vector{Matrix{Float64}}, weights::Union{Vector{Float64}, Nothing}=nothing)
    rank = size(factors[1], 2)
    tensor_shape = [size(f, 1) for f in factors]
    weights = weights === nothing ? ones(rank) : weights
    tensor = zeros(tensor_shape...)
    
    for r in 1:rank
        # Outer product of r-th column from each factor
        component = factors[1][:, r]
        for f in factors[2:end]
            component = vec(component * f[:, r]')
        end
        tensor .+= weights[r] .* reshape(component, tensor_shape...)
    end
    
    return tensor
end

"""
CP decomposition using Alternating Least Squares (ALS).

"""
function als_cp_decomposition(tensor::Array, rank::Int; 
                             max_iter::Int=1000, tol::Float64=1e-6,
                             init::Symbol=:random, rng::AbstractRNG=MersenneTwister(42),
                             verbose::Bool=false)
    tensor_shape = size(tensor)
    n_modes = length(tensor_shape)
    tensor_norm = norm(tensor)
    tensor_norm < 1e-10 && error("Tensor norm too small")
    
    factors = initialize_factors(tensor_shape, rank; init=init, rng=rng)
    prev_error = Inf
    weights = ones(rank) #Initialize weights
    relative_error = 0.0
    final_iteration = max_iter # Default to max_iter if not converged
    
    for iteration in 1:max_iter
        for mode in 1:n_modes
            tensor_unfolded = unfold_tensor(tensor, mode)
            
            # Compute Khatri-Rao product of all other factors
            other_factors = [factors[i] for i in 1:n_modes if i != mode]
            khatri_rao = khatri_rao_product(other_factors)
            
            # Solve: tensor_unfolded ≈ factors[mode] * khatri_rao'
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
                (norm_val = norm(factors[mode][:, r])) > 1e-10 && 
                    (weights[r] = norm_val; factors[mode][:, r] ./= norm_val)
            end
        end
        
        reconstruction = cp_reconstruct(factors, weights)
        relative_error = norm(tensor - reconstruction) / tensor_norm
        
        error_change = prev_error > 1e-10 && !isinf(prev_error) ? 
                      abs(prev_error - relative_error) / prev_error : 
                      (isinf(prev_error) ? 1.0 : abs(relative_error))
        
        if error_change < tol
            final_iteration = iteration
            verbose && println("Converged at iteration $iteration")
            break
        end
        
        prev_error = relative_error
        verbose && iteration % 100 == 0 && 
            @printf("Iteration %d: relative error = %.6f\n", iteration, relative_error)
    end
    
    return Dict(
        :factors => factors,
        :weights => weights,
        :reconstruction => cp_reconstruct(factors, weights),
        :relative_error => relative_error,
        :n_iter => final_iteration,
        :converged => final_iteration < max_iter
    )
end
