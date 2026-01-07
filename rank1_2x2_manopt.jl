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

    problem = Problem(M; cost=cost, grad=rgrad)
    solver = GradientDescent(; linesearch=ArmijoLinesearch(M))
    options = Options(; maxiter=maxiter, stop_when=StopWhenGradientNormLess(tol))

    result = solve(solver, problem, x0; options=options)
    x_star = get_point(result)

    println("Initial point (rank-1):\n", x0)
    println("Optimized point (rank-1):\n", x_star)
    println("Target:\n", TARGET)
    println("Final cost: ", cost(x_star))
    println("Rank of solution: ", rank(x_star))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
