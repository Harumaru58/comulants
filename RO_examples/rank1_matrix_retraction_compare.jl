# Compare two retractions for rank-1 matrix approximation
# 1) Product-manifold retraction (normalize u, v)
# 2) SVD truncation retraction (project back to rank-1)
# 3) Baseline: no normalization on u, v

using LinearAlgebra
using Random
using Manifolds
using Manopt
using RecursiveArrayTools

# ----------------------------
# Problem data
# ----------------------------
m, n = 30, 20
Random.seed!(42)
A = randn(m, n)

# Toggle for baseline variants
const run_no_normalization= true
const run_simple_normalization = true

# ----------------------------
# Product-manifold formulation (Manopt)
# M = R × S^{m-1} × S^{n-1}
# ----------------------------
M = ProductManifold(Euclidean(1), Sphere(m - 1), Sphere(n - 1))

function rank1_matrix(p)
    λ = p.x[1][1]
    u = p.x[2]
    v = p.x[3]
    return λ * u * v'
end

function cost(M, p)
    X = rank1_matrix(p)
    return norm(A - X)^2
end

function egrad(M, p)
    λ = p.x[1][1]
    u = p.x[2]
    v = p.x[3]

    X = λ * u * v'
    G = -2 * (A - X)

    dλ = sum(G .* (u * v'))
    du = λ * (G * v)
    dv = λ * (G' * u)

    return ArrayPartition([dλ], du, dv)
end

function optimize_product_manifold(; stepsize=0.1, maxiter=500, tol=1e-8)
    p0 = ArrayPartition(
        [1.0],
        normalize(randn(m)),
        normalize(randn(n))
    )

    p_opt = gradient_descent(
        M,
        cost,
        egrad,
        p0;
        stepsize=Manopt.ConstantStepsize(M, stepsize),
        stopping_criterion=StopWhenAny(
            StopAfterIteration(maxiter),
            StopWhenGradientNormLess(tol)
        )
    )

    return p_opt
end

# ----------------------------
# SVD-retraction formulation on rank-1 matrices
# ----------------------------
cost_matrix(X) = 0.5 * norm(X - A)^2
agrad(X) = X - A

function project_to_tangent(X, G)
    U, S, V = svd(X)
    u = U[:, 1]
    v = V[:, 1]
    P_u = u * u'
    P_v = v * v'
    return P_u * G + G * P_v - P_u * G * P_v
end

function retract_rank1(Y)
    U, S, V = svd(Y)
    return U[:, 1] * S[1] * V[:, 1]'
end

function optimize_svd_retraction(; stepsize=0.2, maxiter=500, tol=1e-8)
    X = retract_rank1(randn(m, n))
    gnorm = Inf

    for _ in 1:maxiter
        G = agrad(X)
        Rgrad = project_to_tangent(X, G)
        gnorm = norm(Rgrad)
        gnorm < tol && break
        X = retract_rank1(X - stepsize * Rgrad)
    end

    return X, gnorm
end

# ----------------------------
# Baseline: unconstrained (no normalization on u, v)
# ----------------------------
function backtracking_step_no_norm(λ, u, v, dλ, du, dv; stepsize=0.01, c=1e-4, tau=0.5, max_ls=20)
    f0 = norm(A - λ * u * v')^2
    gnorm2 = dλ^2 + norm(du)^2 + norm(dv)^2
    step = stepsize

    for _ in 1:max_ls
        λ_new = λ - step * dλ
        u_new = u - step * du
        v_new = v - step * dv

        # Gauge-fix to keep parameters well-scaled without changing X
        su = norm(u_new)
        sv = norm(v_new)
        if su > 0 && sv > 0
            λ_new *= su * sv
            u_new /= su
            v_new /= sv
        end

        f1 = norm(A - λ_new * u_new * v_new')^2
        if f1 <= f0 - c * step * gnorm2
            return λ_new, u_new, v_new, step
        end
        step *= tau
    end

    return λ, u, v, 0.0
end

function optimize_no_normalization(; stepsize=0.01, maxiter=2000, tol=1e-8)
    λ = 1.0
    u = randn(m)
    v = randn(n)
    gnorm = Inf

    for _ in 1:maxiter
        X = λ * u * v'
        G = -2 * (A - X)

        dλ = sum(G .* (u * v'))
        du = λ * (G * v)
        dv = λ * (G' * u)

        gnorm = sqrt(dλ^2 + norm(du)^2 + norm(dv)^2)
        gnorm < tol && break

        λ, u, v, _ = backtracking_step_no_norm(
            λ, u, v, dλ, du, dv; stepsize=stepsize
        )
    end

    return λ, u, v, gnorm
end

# ----------------------------
# Baseline: normalize u, v after each step (no tangent projection)
# ----------------------------
function backtracking_step_simple_norm(λ, u, v, dλ, du, dv; stepsize=0.01, c=1e-4, tau=0.5, max_ls=20)
    f0 = norm(A - λ * u * v')^2
    gnorm2 = dλ^2 + norm(du)^2 + norm(dv)^2
    step = stepsize

    for _ in 1:max_ls
        λ_new = λ - step * dλ
        u_new = normalize(u - step * du)
        v_new = normalize(v - step * dv)

        f1 = norm(A - λ_new * u_new * v_new')^2
        if f1 <= f0 - c * step * gnorm2
            return λ_new, u_new, v_new, step
        end
        step *= tau
    end

    return λ, u, v, 0.0
end

function optimize_simple_normalization(; stepsize=0.01, maxiter=2000, tol=1e-8)
    λ = 1.0
    u = normalize(randn(m))
    v = normalize(randn(n))
    gnorm = Inf

    for _ in 1:maxiter
        X = λ * u * v'
        G = -2 * (A - X)

        dλ = sum(G .* (u * v'))
        du = λ * (G * v)
        dv = λ * (G' * u)

        # Remove radial components before re-normalizing
        du = du - dot(u, du) * u
        dv = dv - dot(v, dv) * v

        gnorm = sqrt(dλ^2 + norm(du)^2 + norm(dv)^2)
        gnorm < tol && break

        λ, u, v, _ = backtracking_step_simple_norm(
            λ, u, v, dλ, du, dv; stepsize=stepsize
        )
    end

    return λ, u, v, gnorm
end

# ----------------------------
# Run both methods
# ----------------------------
println("Starting rank-1 approximation for A ∈ R^($m×$n)")
println()

p_opt = optimize_product_manifold()
λ_opt = p_opt.x[1][1]
u_opt = p_opt.x[2]
v_opt = p_opt.x[3]
X_prod = λ_opt * u_opt * v_opt'

X_svd_ret, gnorm_svd = optimize_svd_retraction()

λ_none = 0.0
u_none = zeros(m)
v_none = zeros(n)
gnorm_none = NaN
if run_no_normalization
    λ_none, u_none, v_none, gnorm_none = optimize_no_normalization()
end

λ_norm = 0.0
u_norm = zeros(m)
v_norm = zeros(n)
gnorm_norm = NaN
if run_simple_normalization
    λ_norm, u_norm, v_norm, gnorm_norm = optimize_simple_normalization()
end

# Best rank-1 via SVD (closed form)
U, S, V = svd(A)
X_best = S[1] * U[:, 1] * V[:, 1]'

# ----------------------------
# Results
# ----------------------------
println("Product-manifold retraction (normalize u, v)")
println("  Final cost: ", cost(M, p_opt))
println("  Error ||A - X||: ", norm(A - X_prod))
println()
println("SVD retraction (truncate to rank 1)")
println("  Final cost: ", cost_matrix(X_svd_ret))
println("  Error ||A - X||: ", norm(A - X_svd_ret))
println("  Riemannian grad norm: ", gnorm_svd)
println()
if run_no_normalization
    X_none = λ_none * u_none * v_none'
    println("No normalization baseline (u, v unconstrained + gauge fix)")
    println("  Final cost: ", norm(A - X_none)^2)
    println("  Error ||A - X||: ", norm(A - X_none))
    println("  Euclidean grad norm: ", gnorm_none)
    println()
end
if run_simple_normalization
    X_norm = λ_norm * u_norm * v_norm'
    println("Simple normalization baseline (tangent projected)")
    println("  Final cost: ", norm(A - X_norm)^2)
    println("  Error ||A - X||: ", norm(A - X_norm))
    println("  Euclidean grad norm: ", gnorm_norm)
    println()
end
println("Closed-form SVD solution")
println("  Error ||A - X_best||: ", norm(A - X_best))
println()
println("Relative gap to best (product): ",
    abs(norm(A - X_prod) - norm(A - X_best)) / norm(A - X_best)
)
println("Relative gap to best (svd retract): ",
    abs(norm(A - X_svd_ret) - norm(A - X_best)) / norm(A - X_best)
)
if run_no_normalization
    X_none = λ_none * u_none * v_none'
    println("Relative gap to best (no norm): ",
        abs(norm(A - X_none) - norm(A - X_best)) / norm(A - X_best)
    )
end
if run_simple_normalization
    X_norm = λ_norm * u_norm * v_norm'
    println("Relative gap to best (simple norm): ",
        abs(norm(A - X_norm) - norm(A - X_best)) / norm(A - X_best)
    )
end
println("="^70)
