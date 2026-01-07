# Rank-1 2×2 Riemannian gradient descent (explicit tangent + retraction)
#
# Manifold: M = { X ∈ R^{2×2} : rank(X) = 1 }
# Tangent space at X = uσvᵀ (σ>0):
#   T_X M = { u aᵀ + b vᵀ : a,b ∈ R² }
# Projection of an ambient direction G onto T_X M (fixed-rank projection):
#   P_T(G) = P_u G + G P_v - P_u G P_v
#   with P_u = u uᵀ, P_v = v vᵀ
# Retraction (rank-1 SVD truncation):
#   R_X(ξ) = U[:,1] * S[1] * V[:,1]ᵀ  where U,S,V = svd(X+ξ)
# Objective: f(X) = 1/2 ||X - A||_F². Euclidean grad: ∇f = X - A.

using LinearAlgebra
using Random

# Target matrix A
const A = [3.0 1.0; 2.0 0.5]
# Ambient cost and gradient
cost(X) = 0.5 * norm(X - A)^2
agrad(X) = X - A

"""
Compute SVD-based u,v for a rank-1 point X; returns u,v (unit), sigma.
"""
function leading_uv(X)
    U, S, V = svd(X)
    return U[:, 1], V[:, 1], S[1]
end

"""
Project ambient matrix G to tangent space at rank-1 X.
T_X M = { u aᵀ + b vᵀ }. Projection formula: P_u G + G P_v - P_u G P_v.
"""
function project_to_tangent(X, G)
    u, v, _ = leading_uv(X)
    P_u = u * u'
    P_v = v * v'
    return P_u * G + G * P_v - P_u * G * P_v
end

"""
Retraction: truncated SVD back to rank 1.
"""
function retract_rank1(Y)
    U, S, V = svd(Y)
    return U[:, 1] * S[1] * V[:, 1]'
end

"""
Riemannian gradient descent on rank-1 manifold.
"""
function optimize_rank1(; step=0.2, maxiter=500, tol=1e-8, seed=42, verbose=false)
    Random.seed!(seed)
    X = retract_rank1(randn(2, 2))

    for k in 1:maxiter
        G = agrad(X)
        Rgrad = project_to_tangent(X, G)
        gnorm = norm(Rgrad)
        verbose && println("iter=$k grad=$gnorm cost=$(cost(X))")
        if gnorm < tol
            return X, k, gnorm
        end
        X = retract_rank1(X - step * Rgrad)
    end
    return X, maxiter, norm(project_to_tangent(X, agrad(X)))
end

function main()
    X, iters, gnorm = optimize_rank1()
    println("Iterations: $iters")
    println("Riemannian grad norm: $gnorm")
    println("Rank-1 approximation:\n", X)
    println("Target:\n", A)
    println("Final cost: ", cost(X))
    println("Rank of solution: ", rank(X))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
