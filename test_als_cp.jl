"""
Comprehensive test suite for ALS CP Decomposition

This script validates that the ALS implementation:
1. Recovers known factors from synthetic data
2. Monotonic error decrease (optimization works)
3. Convergence behavior
4. Handles different tensor shapes and ranks
"""

using LinearAlgebra
using Random
using Printf
include("als_cp_decomposition.jl")

println("=" ^ 70)
println("ALS CP Decomposition Validation Tests")
println("=" ^ 70)
println()

# Test 1: Recovery of Known Factors (No Noise)
println("Test 1: Recovery of Known Factors (No Noise)")
println("-" ^ 70)
Random.seed!(42)

I, J, K = 10, 15, 20
rank = 3

# Create known factors
factors_true = [
    randn(I, rank),
    randn(J, rank),
    randn(K, rank)
]
weights_true = [2.0, 1.5, 1.0]

# Normalize true factors
for i in 1:length(factors_true)
    for r in 1:rank
        norm_val = norm(factors_true[i][:, r])
        if norm_val > 1e-10
            factors_true[i][:, r] ./= norm_val
        end
    end
end

# Create tensor from known factors
tensor = cp_reconstruct(factors_true, weights_true)

# Decompose
result = als_cp_decomposition(tensor, rank; max_iter=1000, tol=1e-8, 
                              init=:svd, verbose=false)

# Check reconstruction error
reconstruction_error = result[:relative_error]
println("  Reconstruction error: $(@sprintf("%.2e", reconstruction_error))")

if reconstruction_error < 1e-6
    println("  ✓ PASS: Very low reconstruction error")
else
    println("  ✗ FAIL: Reconstruction error too high")
end

# Check if we can recover factors (up to permutation and scaling)
# Compute correlation between recovered and true factors
max_corrs = zeros(rank)
for r in 1:rank
    for r_true in 1:rank
        # Check correlation across all modes
        corr_avg = 0.0
        for mode in 1:3
            corr = abs(cor(factors_true[mode][:, r_true], result[:factors][mode][:, r]))
            corr_avg += corr
        end
        corr_avg /= 3
        max_corrs[r] = max(max_corrs[r], corr_avg)
    end
end

mean_corr = mean(max_corrs)
println("  Mean factor correlation: $(@sprintf("%.4f", mean_corr))")
if mean_corr > 0.95
    println("  ✓ PASS: Factors recovered with high correlation")
else
    println("  ⚠ WARNING: Factor recovery correlation is moderate")
end
println()

# Test 2: Monotonic Error Decrease
println("Test 2: Monotonic Error Decrease (Optimization Progress)")
println("-" ^ 70)

# Modified version to track error history
function als_with_history(tensor::Array, rank::Int; max_iter::Int=100, tol::Float64=1e-6)
    tensor_shape = size(tensor)
    n_modes = length(tensor_shape)
    tensor_norm = norm(tensor)
    
    factors = initialize_factors(tensor_shape, rank; init=:svd, rng=MersenneTwister(42))
    prev_error = Inf
    weights = ones(rank)
    error_history = Float64[]
    
    for iteration in 1:max_iter
        for mode in 1:n_modes
            tensor_unfolded = unfold_tensor(tensor, mode)
            other_factors = [factors[i] for i in 1:n_modes if i != mode]
            khatri_rao = khatri_rao_product(other_factors)
            gram = khatri_rao' * khatri_rao
            rhs = tensor_unfolded * khatri_rao
            try
                factors[mode] = rhs * pinv(gram)
            catch
                factors[mode] = tensor_unfolded * pinv(khatri_rao')
            end
            
            for r in 1:rank
                norm_val = norm(factors[mode][:, r])
                if norm_val > 1e-10
                    weights[r] = norm_val
                    factors[mode][:, r] ./= norm_val
                end
            end
        end
        
        reconstruction = cp_reconstruct(factors, weights)
        error_val = norm(tensor - reconstruction)
        relative_error = error_val / tensor_norm
        push!(error_history, relative_error)
        
        if prev_error > 1e-10 && !isinf(prev_error)
            error_change = abs(prev_error - relative_error) / prev_error
            if error_change < tol
                break
            end
        end
        prev_error = relative_error
    end
    
    return error_history
end

error_history = als_with_history(tensor, rank; max_iter=50, tol=1e-8)

# Check monotonicity (allow small numerical fluctuations)
non_decreasing = true
for i in 2:length(error_history)
    if error_history[i] > error_history[i-1] + 1e-10  # Allow small numerical errors
        non_decreasing = false
        println("  ✗ FAIL: Error increased at iteration $i")
        println("    Error[$i-1] = $(@sprintf("%.6e", error_history[i-1]))")
        println("    Error[$i]   = $(@sprintf("%.6e", error_history[i]))")
        break
    end
end

if non_decreasing
    println("  ✓ PASS: Error decreases monotonically")
    println("  Initial error: $(@sprintf("%.6e", error_history[1]))")
    println("  Final error:   $(@sprintf("%.6e", error_history[end]))")
    println("  Reduction:     $(@sprintf("%.2f", (1 - error_history[end]/error_history[1])*100))%")
end
println()

# Test 3: Convergence Behavior
println("Test 3: Convergence Behavior")
println("-" ^ 70)

# Test with different tolerances
tolerances = [1e-4, 1e-6, 1e-8]
for tol in tolerances
    local result = als_cp_decomposition(tensor, rank; max_iter=1000, tol=tol, 
                                  init=:svd, verbose=false)
    println("  Tolerance: $(@sprintf("%.0e", tol)) -> Iterations: $(result[:n_iter]), Converged: $(result[:converged])")
end
println()

# Test 4: Noisy Data Recovery
println("Test 4: Noisy Data Recovery")
println("-" ^ 70)

noise_levels = [0.01, 0.05, 0.1, 0.2]
for noise_level in noise_levels
    tensor_noisy = tensor .+ noise_level .* randn(size(tensor)...) .* std(tensor)
    local result = als_cp_decomposition(tensor_noisy, rank; max_iter=1000, tol=1e-6, 
                                  init=:svd, verbose=false)
    
    # Compare with true tensor (not noisy)
    true_error = norm(tensor - result[:reconstruction]) / norm(tensor)
    noisy_error = result[:relative_error]
    
    println("  Noise level: $(@sprintf("%.2f", noise_level))")
    println("    Reconstruction error: $(@sprintf("%.4f", noisy_error))")
    println("    Error vs true tensor: $(@sprintf("%.4f", true_error))")
    
    if true_error < noise_level * 2  # Should be close to noise level
        println("    ✓ PASS: Error reasonable for noise level")
    else
        println("    ⚠ WARNING: Error higher than expected")
    end
end
println()

# Test 5: Different Tensor Shapes and Ranks
println("Test 5: Different Tensor Shapes and Ranks")
println("-" ^ 70)

test_cases = [
    ((5, 6, 7), 2),
    ((8, 10, 12), 3),
    ((10, 15, 20), 5),
    ((5, 5, 5, 5), 2),  # 4-way tensor
]

for (shape, test_rank) in test_cases
    # Create synthetic tensor
    factors_test = [randn(s, test_rank) for s in shape]
    weights_test = ones(test_rank)
    
    # Normalize
    for f in factors_test
        for r in 1:test_rank
            norm_val = norm(f[:, r])
            if norm_val > 1e-10
                f[:, r] ./= norm_val
            end
        end
    end
    
    tensor_test = cp_reconstruct(factors_test, weights_test)
    
    local result = als_cp_decomposition(tensor_test, test_rank; max_iter=500, tol=1e-6, 
                                  init=:svd, verbose=false)
    
    println("  Shape: $shape, Rank: $test_rank")
    println("    Error: $(@sprintf("%.2e", result[:relative_error]))")
    println("    Converged: $(result[:converged])")
    
    if result[:relative_error] < 1e-5 && result[:converged]
        println("    ✓ PASS")
    else
        println("    ⚠ WARNING: High error or did not converge")
    end
end
println()

# Test 6: Rank Mismatch (Over/Under-estimation)
println("Test 6: Rank Mismatch Analysis")
println("-" ^ 70)

true_rank = 3
tensor_rank3 = cp_reconstruct(factors_true, weights_true)

for test_rank in [2, 3, 4, 5]
    local result = als_cp_decomposition(tensor_rank3, test_rank; max_iter=1000, tol=1e-6, 
                                  init=:svd, verbose=false)
    
    println("  True rank: $true_rank, Test rank: $test_rank")
    println("    Error: $(@sprintf("%.4f", result[:relative_error]))")
    
    if test_rank < true_rank
        if result[:relative_error] > 0.1
            println("    ✓ PASS: Higher error expected for under-estimated rank")
        else
            println("    ⚠ WARNING: Lower error than expected")
        end
    elseif test_rank == true_rank
        if result[:relative_error] < 1e-5
            println("    ✓ PASS: Low error for correct rank")
        end
    else
        if result[:relative_error] < 1e-5
            println("    ✓ PASS: Low error (over-parameterization)")
        end
    end
end
println()

# Test 7: Initialization Comparison
println("Test 7: Initialization Method Comparison")
println("-" ^ 70)

for init_method in [:random, :svd]
    local result = als_cp_decomposition(tensor, rank; max_iter=1000, tol=1e-6, 
                                  init=init_method, rng=MersenneTwister(42), verbose=false)
    
    println("  Initialization: $init_method")
    println("    Error: $(@sprintf("%.2e", result[:relative_error]))")
    println("    Iterations: $(result[:n_iter])")
    println("    Converged: $(result[:converged])")
end
println()

# Summary
println("=" ^ 70)
println("Test Summary")
println("=" ^ 70)
println("If all tests show ✓ PASS, the ALS implementation is working correctly.")
println("Key indicators of proper optimization:")
println("  1. Low reconstruction error on noise-free data (< 1e-6)")
println("  2. Monotonic error decrease during iterations")
println("  3. Convergence within reasonable iterations")
println("  4. Reasonable error scaling with noise level")
println("  5. Works across different tensor shapes and ranks")
println()

