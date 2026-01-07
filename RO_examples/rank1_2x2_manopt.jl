# Rank-1 2x2 matrix optimization on a fixed-rank manifold using Manopt.jl

using LinearAlgebra
using Random
using Manifolds
using Manopt

# Target full-rank 2x2 matrix we want to approximate with rank 1
const TARGET = [3.0 1.0; 2.0 0.5]

# Manifold of 2x2 matrices with fixed rank 1
typealias = FixedRankMatrices(2, 2, 1)
const M = typealias

# Cost: squared Frobenius distance to TARGET
cost(X) = 0.5 * norm(X - TARGET)^2

# Euclidean gradient
function egrad(X)
    return X - TARGET
end

# Riemannian gradient from Euclidean gradient
function rgrad(X)
    return egrad_to_rgrad(M, X, egrad(X))
end

function main(; seed=42, maxiter=200, tol=1e-8)
    Random.seed!(seed)
    x0 = rand(M)

    # Run gradient descent - Manopt handles the optimization internally
    result = gradient_descent(
        M, cost, rgrad, x0;
        stopping_criterion=StopWhenAny(
            StopAfterIteration(maxiter),
            StopWhenGradientNormLess(tol)
        ),
        debug=[:Iteration, " ", :Cost, "\n", 1]
    )

    println("\nInitial point (rank-1):\n", to_matrix(x0))
    println("Optimized point (rank-1):\n", to_matrix(result))
    println("Target:\n", TARGET)
    println("Final cost: ", cost(x_star))
    println("Rank of solution: ", rank(x_star))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
