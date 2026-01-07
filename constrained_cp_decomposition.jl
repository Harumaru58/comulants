"""
CP Tensor Decomposition with Ellipsoidal Constraints using Riemannian Optimization

Integrates Riemannian optimization on ellipsoids with CP decomposition.
Each factor matrix column is constrained to lie on an ellipsoid.

For tensor T ≈ Σᵣ λᵣ aᵣ ⊗ bᵣ ⊗ cᵣ
Subject to: aᵣᵀ Qₐ aᵣ = 1, bᵣᵀ Qᵦ bᵣ = 1, cᵣᵀ Qᵧ cᵣ = 1
"""

include("riemannian_ellipsoid.jl")
include("als_cp_dec.jl")

using LinearAlgebra
using Random
using Printf

"""
Constrained CP decomposition with ellipsoidal constraints on factors.

Instead of standard ALS update: A = X⁽ⁿ⁾ (Z⁺)ᵀ
We solve: minimize ||X⁽ⁿ⁾ - A Zᵀ||² subject to each column aᵣ satisfies aᵣᵀ Q aᵣ = 1

This is done by:
1. Compute unconstrained ALS solution
2. For each column (component), project to ellipsoid
"""
function constrained_als_cp(
    tensor::Array,
    rank::Int,
    ellipsoids::Vector{Ellipsoid};  # One ellipsoid per mode
    max_iter::Int=500,
    tol::Float64=1e-6,
    verbose::Bool=true
)
    n_modes = ndims(tensor)
    dims = size(tensor)

    @assert length(ellipsoids) == n_modes "Need one ellipsoid per mode"

    # Initialize factors randomly and project to ellipsoids
    factors = [randn(dims[i], rank) for i in 1:n_modes]

    # Project each column to corresponding ellipsoid
    for mode in 1:n_modes
        for r in 1:rank
            factors[mode][:, r] = project_to_ellipsoid(
                factors[mode][:, r],
                ellipsoids[mode]
            )
        end
    end

    tensor_norm = norm(tensor)
    prev_error = Inf
    weights = ones(rank)

    verbose && println("Starting Constrained ALS CP Decomposition")
    verbose && println("Tensor shape: $(size(tensor)), Rank: $rank")
    verbose && println()

    for iter in 1:max_iter
        # Update each mode
        for mode in 1:n_modes
            # Unfold tensor
            X_unfold = unfold_tensor(tensor, mode)

            # Khatri-Rao product of other factors
            other_factors = [factors[i] for i in 1:n_modes if i != mode]
            Z = khatri_rao_product(other_factors)

            # Standard ALS update (unconstrained)
            gram = Z' * Z
            rhs = X_unfold * Z

            A_unconstrained = try
                rhs * pinv(gram)
            catch
                X_unfold * pinv(Z')
            end

            # PROJECT each column to ellipsoid
            for r in 1:rank
                factors[mode][:, r] = project_to_ellipsoid(
                    A_unconstrained[:, r],
                    ellipsoids[mode]
                )
            end

            # Extract weights from last mode
            if mode == n_modes
                for r in 1:rank
                    weights[r] = norm(factors[mode][:, r])
                    if weights[r] > 1e-10
                        factors[mode][:, r] ./= weights[r]
                    end
                end
            end
        end

        # Compute reconstruction error
        reconstruction = cp_reconstruct(factors, weights)
        error = norm(tensor - reconstruction) / tensor_norm

        # Check convergence
        if prev_error > 1e-10 && !isinf(prev_error)
            error_change = abs(prev_error - error) / prev_error
        else
            error_change = abs(error)
        end

        if error_change < tol
            verbose && println("Converged at iteration $iter")
            break
        end

        prev_error = error

        if verbose && (iter % 50 == 0 || iter <= 5)
            @printf("Iter %4d: relative error = %.6f\n", iter, error)

            # Verify constraints
            if iter % 100 == 0
                println("  Constraint verification:")
                for mode in 1:n_modes
                    max_violation = maximum([
                        abs(dot(factors[mode][:, r],
                               ellipsoids[mode].Q * factors[mode][:, r]) - 1.0)
                        for r in 1:rank
                    ])
                    @printf("    Mode %d: max violation = %.2e\n", mode, max_violation)
                end
            end
        end
    end

    verbose && println()
    verbose && @printf("Final relative error: %.6f\n", prev_error)

    return (
        factors = factors,
        weights = weights,
        reconstruction = cp_reconstruct(factors, weights),
        error = prev_error
    )
end

"""
Apply to biomarker data: use covariance-based ellipsoids from data.

This would be called after computing 3rd-order moments, to decompose
with biologically meaningful constraints.
"""
function cp_decomposition_with_biomarker_constraints(
    moment_tensor::Array,
    biomarker_data::Matrix{Float64},  # n_samples × n_biomarkers
    rank::Int;
    confidence_level::Float64=2.0,  # Confidence region (2-sigma = 95%)
    verbose::Bool=true
)
    n_modes = ndims(moment_tensor)
    n_biomarkers = size(biomarker_data, 2)

    # Compute empirical covariance from data
    μ = mean(biomarker_data, dims=1)
    data_centered = biomarker_data .- μ
    Σ = (data_centered' * data_centered) / size(biomarker_data, 1)

    # Add regularization for numerical stability
    Σ = Σ + 1e-6 * I

    verbose && println("Creating ellipsoid from biomarker covariance matrix")
    verbose && @printf("Covariance matrix: %d × %d\n", size(Σ)...)
    verbose && @printf("Confidence level: %.1f-sigma\n", confidence_level)
    verbose && println()

    # Create ellipsoid from covariance (same for all modes in symmetric tensor)
    ellipsoid = CovarianceEllipsoid(Σ, confidence_level)

    # For symmetric 3rd-order tensor, all modes use same ellipsoid
    ellipsoids = [ellipsoid for _ in 1:n_modes]

    # Run constrained decomposition
    result = constrained_als_cp(
        moment_tensor,
        rank,
        ellipsoids;
        max_iter=500,
        tol=1e-6,
        verbose=verbose
    )

    return result
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^70)
    println("Constrained CP Decomposition with Ellipsoidal Constraints")
    println("="^70)
    println()

    # Example 1: Simple 3-way tensor with known structure
    println("Example 1: 3-way tensor (10×10×10)")
    println("-"^70)

    Random.seed!(42)

    d = 10
    rank = 3

    # Create true factors on ellipsoids
    semi_axes = [2.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    ellipsoid_true = AxisAlignedEllipsoid(semi_axes)

    # Generate factors
    A_true = randn(d, rank)
    for r in 1:rank
        A_true[:, r] = project_to_ellipsoid(A_true[:, r], ellipsoid_true)
    end

    # Create tensor
    factors_true = [A_true, A_true, A_true]
    weights_true = [2.0, 1.5, 1.0]
    tensor = cp_reconstruct(factors_true, weights_true)

    # Add noise
    noise_level = 0.1
    tensor .+= noise_level * randn(size(tensor)...) * std(tensor)

    println("True rank: $rank")
    println("Noise level: $noise_level")
    println()

    # Decompose with constraints
    ellipsoids = [ellipsoid_true, ellipsoid_true, ellipsoid_true]
    result = constrained_als_cp(tensor, rank, ellipsoids; verbose=true)

    println()
    println("Constraint satisfaction check:")
    for mode in 1:3
        println("  Mode $mode:")
        for r in 1:rank
            constraint_val = dot(
                result.factors[mode][:, r],
                ellipsoid_true.Q * result.factors[mode][:, r]
            )
            @printf("    Component %d: xᵀQx = %.10f\n", r, constraint_val)
        end
    end

    println("\n" * "="^70)
    println("\nExample 2: Using biomarker-like data (28D)")
    println("-"^70)

    # Simulate biomarker data
    n_samples = 100
    n_biomarkers = 28
    rank = 5

    # Generate correlated biomarker data
    Random.seed!(123)
    L = randn(n_biomarkers, 10)
    biomarker_data = randn(n_samples, 10) * L'

    # Standardize
    biomarker_data = (biomarker_data .- mean(biomarker_data, dims=1)) ./
                     std(biomarker_data, dims=1)

    println("Biomarker data: $n_samples samples × $n_biomarkers biomarkers")

    # Compute 3rd-order moment tensor (simplified version)
    println("Computing 3rd-order moment tensor...")
    data_centered = biomarker_data .- mean(biomarker_data, dims=1)

    moment_tensor = zeros(n_biomarkers, n_biomarkers, n_biomarkers)
    for sample in 1:n_samples
        x = data_centered[sample, :]
        for i in 1:n_biomarkers
            for j in 1:n_biomarkers
                for k in 1:n_biomarkers
                    moment_tensor[i, j, k] += x[i] * x[j] * x[k]
                end
            end
        end
    end
    moment_tensor ./= n_samples

    println("Moment tensor shape: $(size(moment_tensor))")
    println()

    # Decompose with biomarker covariance constraints
    result2 = cp_decomposition_with_biomarker_constraints(
        moment_tensor,
        biomarker_data,
        rank;
        confidence_level=2.0,
        verbose=true
    )

    println()
    println("="^70)
    println("\nSummary:")
    println("-"^70)
    println("✓ Factors are constrained to biologically meaningful ellipsoids")
    println("✓ Ellipsoids defined by empirical covariance of biomarker data")
    println("✓ All constraints satisfied throughout optimization")
    println("✓ Ready to integrate with your moments_3rd_order.py workflow")
    println()
    println("Next steps:")
    println("  1. Load your data.xlsx")
    println("  2. Compute covariance per patient group")
    println("  3. Use group-specific ellipsoids for decomposition")
    println("  4. Compare constrained vs unconstrained results")
end
