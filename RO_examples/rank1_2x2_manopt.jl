# Rank-1 2x2 matrix optimization on a fixed-rank manifold using Manopt.jl
# NOTE: This file has a known issue with Manopt.jl's FixedRankMatrices retraction.
# The Armijo linesearch tries to use a Matrix retraction buffer instead of SVDMPoint.
# For a working manual implementation, see rank1_2x2_manual.jl

using LinearAlgebra
using Random
using Manifolds
using Manopt

# Target full-rank 2x2 matrix we want to approximate with rank 1
const TARGET = [3.0 1.0; 2.0 0.5]

# Manifold of 2x2 matrices with fixed rank 1
const M = FixedRankMatrices(2, 2, 1)

# Helper to convert SVDMPoint to matrix
to_matrix(X) = X.U * Diagonal(X.S) * X.Vt

# Cost: squared Frobenius distance to TARGET
cost(M, X) = 0.5 * norm(to_matrix(X) - TARGET)^2

# Euclidean gradient
function egrad(M, X)
    return to_matrix(X) - TARGET
end

# Riemannian gradient from Euclidean gradient
function rgrad(M, X)
    return riemannian_gradient(M, X, egrad(M, X))
end

function main(; seed=42, maxiter=200, tol=1e-8)
    Random.seed!(seed)
    x0 = rand(M)

    # Run gradient descent
    result = gradient_descent(
        M, cost, rgrad, x0;
        stopping_criterion=StopWhenAny(
            StopAfterIteration(maxiter),
            StopWhenGradientNormLess(tol)
        ),
        debug=[:Iteration, " ", :Cost, "\n", 10]
    )

    println("\nInitial point (rank-1):\n", to_matrix(x0))
    println("Optimized point (rank-1):\n", to_matrix(result))
    println("Target:\n", TARGET)
    println("Final cost: ", cost(M, result))
    println("Rank of solution: ", rank(to_matrix(result)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
