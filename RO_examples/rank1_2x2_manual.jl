# Rank-1 2×2 optimization without Manopt.jl (explicit tangent + retraction)

using LinearAlgebra
using Random

# Target matrix to approximate with rank 1
const TARGET = [3.0 1.0; 2.0 0.5]

# Cost in ambient space
cost(X) = 0.5 * norm(X - TARGET)^2

# Euclidean gradient
egrad(X) = X - TARGET

# Project an ambient matrix to the tangent space at a rank-1 point X = u*s*v'
# Tangent space: { u a' + b v' : a,b in R^2 }
function project_to_tangent(X, G)
    U, S, V = svd(X)
    u = U[:, 1]
    v = V[:, 1]
    P_u = u * u'
    P_v = v * v'
    # Projection formula for fixed-rank manifold
    return P_u * G + G * P_v - P_u * G * P_v
end

# Retraction: truncate SVD back to rank 1
function retract(X)
    U, S, V = svd(X)
    return U[:, 1] * S[1] * V[:, 1]'
end

function optimize_rank1(; step=0.2, maxiter=500, tol=1e-8, seed=42)
    Random.seed!(seed)
    # Start from random rank-1 matrix
    X = retract(randn(2, 2))

    for k in 1:maxiter
        G = egrad(X)
        Rgrad = project_to_tangent(X, G)
        gnorm = norm(Rgrad)
        if gnorm < tol
            return X, k, gnorm
        end
        # Riemannian gradient step + retraction
        X = retract(X - step * Rgrad)
    end
    return X, maxiter, norm(project_to_tangent(X, egrad(X)))
end

function main()
    X, iters, gnorm = optimize_rank1()
    println("Iterations: ", iters)
    println("Riemannian grad norm: ", gnorm)
    println("Rank-1 approximation:\n", X)
    println("Target:\n", TARGET)
    println("Final cost: ", cost(X))
    println("Rank of X: ", rank(X))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
