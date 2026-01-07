"""
Riemannian Optimization on n-dimensional Ellipsoids

Generalizes ellipse optimization to arbitrary dimensions.
Ellipsoid constraint: xᵀ Q x = 1, where Q is positive definite matrix

For axis-aligned ellipsoid: x₁²/a₁² + x₂²/a₂² + ... + xₙ²/aₙ² = 1
Q = diag(1/a₁², 1/a₂², ..., 1/aₙ²)
"""

using LinearAlgebra
using Random
using Printf

"""
Ellipsoid defined by xᵀ Q x = 1
"""
struct Ellipsoid
    Q::Matrix{Float64}      # Positive definite matrix defining ellipsoid
    Q_sqrt::Matrix{Float64} # Matrix square root for projections

    function Ellipsoid(Q::Matrix{Float64})
        # Verify Q is symmetric positive definite
        if !issymmetric(Q)
            error("Q must be symmetric")
        end

        eigs = eigvals(Q)
        if any(eigs .<= 0)
            error("Q must be positive definite")
        end

        # Compute Q^(1/2) for projections
        F = eigen(Q)
        Q_sqrt = F.vectors * Diagonal(sqrt.(F.values)) * F.vectors'

        new(Q, Q_sqrt)
    end
end

"""
Create axis-aligned ellipsoid from semi-axes lengths
"""
function AxisAlignedEllipsoid(semi_axes::Vector{Float64})
    n = length(semi_axes)
    Q = Diagonal(1 ./ (semi_axes .^ 2))
    return Ellipsoid(Matrix(Q))
end

"""
Create ellipsoid from covariance matrix (for statistical applications)
The ellipsoid is: xᵀ Σ⁻¹ x = c² (c-sigma confidence region)
"""
function CovarianceEllipsoid(Σ::Matrix{Float64}, c::Float64=1.0)
    # Ensure Σ is symmetric
    Σ_sym = (Σ + Σ') / 2
    Q = inv(Σ_sym) / c^2
    # Ensure Q is also symmetric (numerical stability)
    Q_sym = (Q + Q') / 2
    return Ellipsoid(Q_sym)
end

"""
Project point x onto ellipsoid xᵀ Q x = 1

Uses the fact that projection minimizes ||x - y||² subject to yᵀ Q y = 1
Solution: y = x / sqrt(xᵀ Q x)
"""
function project_to_ellipsoid(x::Vector{Float64}, ellipsoid::Ellipsoid)
    Q = ellipsoid.Q

    # Compute scaling factor
    scale = sqrt(dot(x, Q * x))

    if scale < 1e-10
        # If x is too close to zero, return a random point on ellipsoid
        x_random = randn(length(x))
        scale = sqrt(dot(x_random, Q * x_random))
        return x_random / scale
    end

    return x / scale
end

"""
Verify if point is on ellipsoid (within tolerance)
"""
function is_on_ellipsoid(x::Vector{Float64}, ellipsoid::Ellipsoid; tol::Float64=1e-8)
    constraint_val = dot(x, ellipsoid.Q * x)
    return abs(constraint_val - 1.0) < tol
end

"""
Project vector v onto tangent space at point x on ellipsoid.

Tangent space is orthogonal to gradient of constraint g(x) = xᵀ Q x - 1
∇g(x) = 2Qx (normal to ellipsoid at x)

Projection: v_tangent = v - (v · n)n where n = Qx / ||Qx||
"""
function project_to_tangent_space(v::Vector{Float64}, x::Vector{Float64},
                                  ellipsoid::Ellipsoid)
    Q = ellipsoid.Q

    # Normal vector: Qx
    normal = Q * x
    normal_norm = norm(normal)

    if normal_norm < 1e-10
        return v
    end

    normal = normal / normal_norm

    # Project v onto tangent space
    v_tangent = v - dot(v, normal) * normal

    return v_tangent
end

"""
Retraction: move from x along direction v, stay on ellipsoid
Using projection-based retraction: retract(x, v) = project(x + v)
"""
function retract(x::Vector{Float64}, v::Vector{Float64}, α::Float64,
                ellipsoid::Ellipsoid)
    x_new = x + α * v
    return project_to_ellipsoid(x_new, ellipsoid)
end

"""
Riemannian gradient descent on ellipsoid

Parameters:
-----------
f : Function
    Objective function f(x) to minimize
grad_f : Function
    Euclidean gradient of f
ellipsoid : Ellipsoid
    Constraint manifold xᵀ Q x = 1
x0 : Vector
    Initial point (will be projected to ellipsoid)
max_iter : Int
    Maximum iterations
α : Float64
    Step size (or use adaptive line search)
tol : Float64
    Convergence tolerance on Riemannian gradient norm
verbose : Bool
    Print progress
"""
function riemannian_gradient_descent(
    f::Function,
    grad_f::Function,
    ellipsoid::Ellipsoid,
    x0::Vector{Float64};
    max_iter::Int=1000,
    α::Float64=0.1,
    tol::Float64=1e-6,
    verbose::Bool=true
)
    n = length(x0)

    # Project initial point to ellipsoid
    x = project_to_ellipsoid(x0, ellipsoid)

    if verbose
        @printf("Dimension: %d\n", n)
        @printf("Initial constraint value: %.10f (should be 1.0)\n",
                dot(x, ellipsoid.Q * x))
        @printf("Initial objective: %.6f\n\n", f(x))
    end

    trajectory = [copy(x)]
    objectives = [f(x)]

    for iter in 1:max_iter
        # Euclidean gradient
        grad_eucl = grad_f(x)

        # Riemannian gradient (project to tangent space)
        grad_riem = project_to_tangent_space(grad_eucl, x, ellipsoid)

        grad_norm = norm(grad_riem)

        # Check convergence
        if grad_norm < tol
            if verbose
                println("Converged at iteration $iter")
            end
            break
        end

        # Retract along negative Riemannian gradient
        x_new = retract(x, -grad_riem, α, ellipsoid)

        # Update
        x = x_new
        push!(trajectory, copy(x))
        push!(objectives, f(x))

        if verbose && (iter % 100 == 0 || iter <= 10)
            constraint_val = dot(x, ellipsoid.Q * x)
            @printf("Iter %4d: f = %.6f, ||grad|| = %.6e, constraint = %.10f\n",
                    iter, f(x), grad_norm, constraint_val)
        end
    end

    if verbose
        println()
        @printf("Final objective: %.6f\n", f(x))
        @printf("Final constraint: %.10f (should be 1.0)\n",
                dot(x, ellipsoid.Q * x))
        @printf("Final gradient norm: %.6e\n",
                norm(project_to_tangent_space(grad_f(x), x, ellipsoid)))
    end

    return (
        optimum = x,
        trajectory = trajectory,
        objectives = objectives,
        converged = norm(project_to_tangent_space(grad_f(x), x, ellipsoid)) < tol
    )
end

"""
Riemannian gradient descent with Armijo line search
"""
function riemannian_gradient_descent_armijo(
    f::Function,
    grad_f::Function,
    ellipsoid::Ellipsoid,
    x0::Vector{Float64};
    max_iter::Int=1000,
    α_init::Float64=1.0,
    β::Float64=0.5,
    σ::Float64=0.1,
    tol::Float64=1e-6,
    verbose::Bool=true
)
    x = project_to_ellipsoid(x0, ellipsoid)

    if verbose
        @printf("Using Armijo line search (α_init=%.2f, β=%.2f, σ=%.2f)\n",
                α_init, β, σ)
        @printf("Initial objective: %.6f\n\n", f(x))
    end

    trajectory = [copy(x)]
    objectives = [f(x)]

    for iter in 1:max_iter
        grad_eucl = grad_f(x)
        grad_riem = project_to_tangent_space(grad_eucl, x, ellipsoid)

        grad_norm = norm(grad_riem)

        if grad_norm < tol
            verbose && println("Converged at iteration $iter")
            break
        end

        # Armijo line search
        α = α_init
        f_current = f(x)
        direction = -grad_riem

        for _ in 1:20  # Max line search iterations
            x_new = retract(x, direction, α, ellipsoid)
            f_new = f(x_new)

            # Armijo condition: sufficient decrease
            if f_new <= f_current + σ * α * dot(grad_riem, direction)
                x = x_new
                break
            end

            α *= β
        end

        push!(trajectory, copy(x))
        push!(objectives, f(x))

        if verbose && (iter % 100 == 0 || iter <= 10)
            @printf("Iter %4d: f = %.6f, ||grad|| = %.6e, α = %.6f\n",
                    iter, f(x), grad_norm, α)
        end
    end

    if verbose
        println()
        @printf("Final objective: %.6f\n", f(x))
        @printf("Final constraint: %.10f\n", dot(x, ellipsoid.Q * x))
    end

    return (optimum = x, trajectory = trajectory, objectives = objectives)
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^70)
    println("Riemannian Optimization on n-dimensional Ellipsoid")
    println("="^70)
    println()

    # Example 1: 5D ellipsoid
    println("Example 1: 5D Ellipsoid")
    println("-"^70)

    n = 5
    semi_axes = [3.0, 2.5, 2.0, 1.5, 1.0]
    ellipsoid = AxisAlignedEllipsoid(semi_axes)

    println("Ellipsoid: x₁²/$(semi_axes[1])² + x₂²/$(semi_axes[2])² + ... = 1")
    println()

    # Objective: minimize quadratic form f(x) = xᵀ A x
    A = Diagonal([1.0, 2.0, 3.0, 4.0, 5.0])
    f(x) = dot(x, A * x)
    grad_f(x) = 2 * A * x

    println("Objective: f(x) = Σᵢ i·xᵢ²")
    println()

    # Random initial point
    Random.seed!(42)
    x0 = randn(n)

    result = riemannian_gradient_descent(
        f, grad_f, ellipsoid, x0;
        max_iter=1000, α=0.1, verbose=true
    )

    println()
    println("Optimal solution:")
    for i in 1:n
        @printf("  x[%d] = %8.4f\n", i, result.optimum[i])
    end

    println("\n" * "="^70)
    println("\nExample 2: 10D Ellipsoid with Armijo line search")
    println("-"^70)

    n = 10
    semi_axes = range(3.0, 1.0, length=n) |> collect
    ellipsoid2 = AxisAlignedEllipsoid(semi_axes)

    # More complex objective: Rosenbrock-like
    function f2(x)
        sum = 0.0
        for i in 1:(length(x)-1)
            sum += 100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2
        end
        return sum
    end

    function grad_f2(x)
        n = length(x)
        g = zeros(n)

        for i in 1:(n-1)
            g[i] += -400*x[i]*(x[i+1] - x[i]^2) - 2*(1 - x[i])
            g[i+1] += 200*(x[i+1] - x[i]^2)
        end

        return g
    end

    x0 = randn(n)

    result2 = riemannian_gradient_descent_armijo(
        f2, grad_f2, ellipsoid2, x0;
        max_iter=1000, α_init=1.0, verbose=true
    )

    println()
    println("="^70)
    println("\nExample 3: Covariance-based ellipsoid (28D - biomarker dimension)")
    println("-"^70)

    n = 28
    Random.seed!(123)

    # Create a covariance matrix (simulating biomarker covariances)
    # In practice, this would come from your data.xlsx
    L = randn(n, Int(n/2))
    Σ = (L * L') / n + I  # Ensure positive definite and symmetric

    ellipsoid3 = CovarianceEllipsoid(Σ, 1.0)

    println("Ellipsoid from 28×28 covariance matrix (simulating biomarkers)")

    # Simple quadratic objective
    f3(x) = sum(x.^2)
    grad_f3(x) = 2*x

    x0 = randn(n)

    result3 = riemannian_gradient_descent(
        f3, grad_f3, ellipsoid3, x0;
        max_iter=500, α=0.1, verbose=true
    )

    println()
    println("This demonstrates optimization in the biomarker dimension!")
    println("Next: integrate with CP tensor decomposition.")
end
