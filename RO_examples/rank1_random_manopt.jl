# Simple Rank-1 Matrix Approximation using Manopt.jl
# Find X = λ * u * v' that minimizes ||A - X||²
# where λ ∈ ℝ, u ∈ S^{m-1}, v ∈ S^{n-1}

using LinearAlgebra
using Manifolds
using Manopt
using RecursiveArrayTools

# ----------------------------
# Problem data
# ----------------------------
m, n = 30, 20
A = randn(m, n)

# ----------------------------
# Manifold: ℝ × S^{m-1} × S^{n-1}
# ----------------------------
M = ProductManifold(
    Euclidean(1),          # λ ∈ ℝ
    Sphere(m - 1),         # u ∈ S^{m-1}
    Sphere(n - 1)          # v ∈ S^{n-1}
)

# ----------------------------
# Embedding map: φ(λ,u,v) = λ u v'
# ----------------------------
function rank1_matrix(p)
    λ = p[1][1]
    u = p[2]
    v = p[3]
    return λ * u * v'
end

# ----------------------------
# Cost: f(p) = ||A - λuv'||²
# ----------------------------
function cost(M, p)
    X = rank1_matrix(p)
    return norm(A - X)^2
end

# ----------------------------
# Euclidean gradient (ambient space)
# ----------------------------
function egrad(M, p)
    λ = p[1][1]
    u = p[2]
    v = p[3]

    X = λ * u * v'
    G = -2 * (A - X)  # ∇_X f = -2(A - X)

    # Pullback via Dφ*: differentiate f ∘ φ
    dλ = sum(G .* (u * v'))  # = ⟨G, uv'⟩
    du = λ * (G * v)
    dv = λ * (G' * u)

    return (
        [dλ],     # Component in ℝ
        du,       # Component in ℝ^m (Manopt will project to tangent space)
        dv        # Component in ℝ^n
    )
end

# ----------------------------
# Initial point (ArrayPartition for ProductManifold)
# ----------------------------
p0 = ArrayPartition(
    [1.0],
    normalize(randn(m)),
    normalize(randn(n))
)

# ----------------------------
# Solve using gradient descent
# ----------------------------
println("Starting optimization...")
println("Target matrix size: $m × $n")
println("Initial cost: ", cost(M, p0))
println()

p_opt = gradient_descent(
    M,
    cost,
    egrad,
    p0;
    stepsize = 0.1,
    stopping_criterion = StopWhenAny(
        StopAfterIteration(500),
        StopWhenGradientNormLess(1e-8)
    ),
    debug = [:Iteration, " | ", :Cost, " | ", :GradientNorm, "\n", 50]
)

# ----------------------------
# Extract solution
# ----------------------------
λ_opt = p_opt.x[1][1]
u_opt = p_opt.x[2]
v_opt = p_opt.x[3]

X_opt = λ_opt * u_opt * v_opt'

# ----------------------------
# Results
# ----------------------------
println()
println("="^70)
println("RESULTS")
println("="^70)
println("Optimal λ: ", round(λ_opt, digits = 4))
println("||u||: ", norm(u_opt))
println("||v||: ", norm(v_opt))
println()
println("Final cost: ", cost(M, p_opt))
println("Approximation error: ", norm(A - X_opt))
println()

# Compare with SVD (ground truth)
U, S, V = svd(A)
X_svd = S[1] * U[:, 1] * V[:, 1]'
println("SVD best rank-1 error: ", norm(A - X_svd))
println(
    "Relative difference: ",
    abs(norm(A - X_opt) - norm(A - X_svd)) / norm(A - X_svd)
)
println("="^70)
