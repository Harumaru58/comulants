# Rank-1 3×3 matrix optimization using Riemannian optimization

using LinearAlgebra
using Random

# Target: rank-2 matrix
const TARGET = [4.0 2.1 1.0;
                3.0 1.5 0.75;
                2.0 1.0 0.5]

cost(X) = 0.5 * norm(X - TARGET)^2
egrad(X) = X - TARGET

# Project gradient to tangent space of rank-1 manifold
function project_to_tangent(X, G)
    U, S, V = svd(X)
    u, v = U[:, 1], V[:, 1]
    P_u, P_v = u * u', v * v'
    return P_u * G + G * P_v - P_u * G * P_v
end

# Retract to rank-1 manifold via SVD truncation
retract(X) = (U, S, V = svd(X); U[:, 1] * S[1] * V[:, 1]')

function optimize_rank1(; step=0.15, maxiter=1000, tol=1e-8, seed=42)
    Random.seed!(seed)
    X = retract(randn(3, 3))
    
    for k in 1:maxiter
        Rgrad = project_to_tangent(X, egrad(X))
        norm(Rgrad) < tol && return X, k, cost(X)
        X = retract(X - step * Rgrad)
    end
    return X, maxiter, cost(X)
end

# Run optimization
X, iters, final_cost = optimize_rank1()

# Display result
println("\nRank-1 Approximation (converged in $iters iterations):")
println("Final cost: $final_cost")
println("\nSolution:")
display(round.(X, digits=4))
println("\n\nTarget:")
display(TARGET)
println("\n\nError:")
display(round.(X - TARGET, digits=4))

U, S, V = svd(X)
println("\n\nOuter product: X = $(round(S[1], digits=3)) × u ⊗ v")
println("u = $(round.(U[:, 1], digits=3))")
println("v = $(round.(V[:, 1], digits=3))")
