# Rank-1 2x2 optimization without Manopt.jl (factor-gradient descent)

using LinearAlgebra
using Random

# Target matrix to approximate with rank-1
const TARGET = [3.0 1.0; 2.0 0.5]

# Cost and gradients in factor form: M = u * v'
cost(u, v) = 0.5 * norm(u * v' - TARGET)^2

grad_u(u, v) = (u * v' - TARGET) * v

grad_v(u, v) = (u * v' - TARGET)' * u

function optimize_rank1(; step=0.05, maxiter=1000, tol=1e-8, seed=42)
    Random.seed!(seed)
    u = randn(2)
    v = randn(2)

    for k in 1:maxiter
        gu = grad_u(u, v)
        gv = grad_v(u, v)
        gnorm = max(norm(gu), norm(gv))
        if gnorm < tol
            return u, v, k
        end
        u -= step * gu
        v -= step * gv
    end
    return u, v, maxiter
end

function main()
    u, v, iters = optimize_rank1()
    X = u * v'
    println("Iterations: ", iters)
    println("u = ", u)
    println("v = ", v)
    println("Rank-1 approximation:\n", X)
    println("Target:\n", TARGET)
    println("Final cost: ", cost(u, v))
    println("Rank of X: ", rank(X))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
