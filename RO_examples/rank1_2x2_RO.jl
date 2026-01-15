using LinearAlgebra
using Random

# Rank-1 matrix: X = lambda u vᵀ, ||u|| = ||v|| = 1
struct Rank1
    lambda::Float64
    u::Vector{Float64}
    v::Vector{Float64}
end

matrix(X::Rank1) = X.lambda * X.u * X.v'

# -------------------------------------------------------------------
# Example intrinsic cost: f(X) = 1/2 ||X||_F^2 = 1/2 lambda^2
# -------------------------------------------------------------------
cost(X::Rank1) = 0.5 * X.lambda^2
egrad(X::Rank1) = matrix(X)

# -------------------------------------------------------------------
# Tangent projection
# Tangent vectors: (α, w_u, w_v) with w_u ⟂ u, w_v ⟂ v
# -------------------------------------------------------------------
function tangent_projection(G, X::Rank1)
    u, v, lambda = X.u, X.v, X.lambda

    α   = dot(u, G * v)
    w_u = (G * v - α * u) / lambda
    w_v = (G' * u - α * v) / lambda

    return α, w_u, w_v
end

# -------------------------------------------------------------------
# First-order retraction (no SVD)
# Normalize u and v, absorb scaling into lambda
# -------------------------------------------------------------------
function retract(X::Rank1, α, w_u, w_v, η)
    lambda_new = X.lambda - η * α

    u_tmp = X.u - η * w_u
    v_tmp = X.v - η * w_v

    u_new = u_tmp / norm(u_tmp)
    v_new = v_tmp / norm(v_tmp)

    return Rank1(lambda_new, u_new, v_new)
end

# -------------------------------------------------------------------
# Retraction without normalization
# Absorb norms into lambda instead of normalizing vectors
# -------------------------------------------------------------------
function retract_unnormalized(X::Rank1, α, w_u, w_v, η)
    lambda_new = X.lambda - η * α

    u_new = X.u - η * w_u
    v_new = X.v - η * w_v

    # Absorb norms into lambda
    lambda_scaled = lambda_new * norm(u_new) * norm(v_new)
    u_normalized = u_new / norm(u_new)
    v_normalized = v_new / norm(v_new)

    return Rank1(lambda_scaled, u_normalized, v_normalized)
end

# -------------------------------------------------------------------
# Riemannian gradient descent
# -------------------------------------------------------------------
function optimize_rank1(; stepsize=0.1, maxiter=200, tol=1e-10, use_unnormalized=false)
    Random.seed!(0)

    u0 = randn(2); u0 /= norm(u0)
    v0 = randn(2); v0 /= norm(v0)
    lambda0 = randn()

    X = Rank1(lambda0, u0, v0)

    retract_fn = use_unnormalized ? retract_unnormalized : retract

    for _ in 1:maxiter
        G = egrad(X)
        α, w_u, w_v = tangent_projection(G, X)

        gnorm = sqrt(α^2 + X.lambda^2 * (norm(w_u)^2 + norm(w_v)^2))
        gnorm < tol && break

        X = retract_fn(X, α, w_u, w_v, stepsize)
    end

    return X
end

X_opt = optimize_rank1()
X_opt_unnorm = optimize_rank1(use_unnormalized=true)
