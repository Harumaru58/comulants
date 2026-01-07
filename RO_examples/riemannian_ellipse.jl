"""
Riemannian Optimization on an Ellipse

Demonstrates optimization on the ellipse manifold:
    x²/a² + y²/b² = 1

Key concepts:
- Projection: mapping arbitrary points to the ellipse
- Tangent space: directions you can move while staying on ellipse
- Riemannian gradient: projection of Euclidean gradient to tangent space
- Retraction: moving along tangent direction while staying on manifold
"""

using LinearAlgebra
using Printf
using Plots

# Ellipse parameters
struct Ellipse
    a::Float64  # semi-major axis
    b::Float64  # semi-minor axis
end

"""
Project a point (x, y) onto the ellipse.
Uses parametric form: (a*cos(θ), b*sin(θ))
"""
function project_to_ellipse(p::Vector{Float64}, ellipse::Ellipse)
    x, y = p
    a, b = ellipse.a, ellipse.b

    # Find angle θ that minimizes distance to p
    # This is done by finding θ where gradient of distance is zero
    θ = atan(b * y, a * x)

    return [a * cos(θ), b * sin(θ)]
end

"""
Compute tangent vector to ellipse at point p.
The tangent space is 1D (since ellipse is 1D manifold in 2D space).

For ellipse x²/a² + y²/b² = 1, the gradient of constraint is:
∇f = [2x/a², 2y/b²]

Tangent space is orthogonal to this gradient.
"""
function tangent_vector(p::Vector{Float64}, ellipse::Ellipse)
    x, y = p
    a, b = ellipse.a, ellipse.b

    # Normal vector to ellipse at p
    normal = [2*x/a^2, 2*y/b^2]

    # Tangent is perpendicular to normal (rotate 90 degrees)
    tangent = [-normal[2], normal[1]]

    # Normalize
    return tangent / norm(tangent)
end

"""
Project a vector v onto the tangent space at point p on the ellipse.
This gives the Riemannian gradient.
"""
function project_to_tangent_space(v::Vector{Float64}, p::Vector{Float64}, ellipse::Ellipse)
    x, y = p
    a, b = ellipse.a, ellipse.b

    # Normal vector (gradient of constraint)
    normal = [2*x/a^2, 2*y/b^2]
    normal = normal / norm(normal)

    # Project v onto tangent space: v - (v · n)n
    v_tangent = v - dot(v, normal) * normal

    return v_tangent
end

"""
Retraction: move from point p along tangent direction v, staying on ellipse.
We use exponential map approximation: project p + α*v back to ellipse.
"""
function retract(p::Vector{Float64}, v::Vector{Float64}, α::Float64, ellipse::Ellipse)
    # Move in Euclidean space
    p_new = p + α * v

    # Project back to ellipse
    return project_to_ellipse(p_new, ellipse)
end

"""
Example objective function: minimize f(x,y) = x² + 2y²
Subject to constraint: x²/a² + y²/b² = 1
"""
function objective_function(p::Vector{Float64})
    x, y = p
    return x^2 + 2*y^2
end

"""
Euclidean gradient of objective function
"""
function gradient_euclidean(p::Vector{Float64})
    x, y = p
    return [2*x, 4*y]
end

"""
Riemannian gradient descent on ellipse
"""
function riemannian_gradient_descent(
    f::Function,           # Objective function
    grad_f::Function,      # Euclidean gradient
    ellipse::Ellipse,
    p0::Vector{Float64};   # Initial point (will be projected)
    max_iter::Int=1000,
    α::Float64=0.1,        # Step size
    tol::Float64=1e-6,
    verbose::Bool=true
)
    # PROJECT initial point to ellipse
    p = project_to_ellipse(p0, ellipse)

    trajectory = [copy(p)]
    objectives = [f(p)]

    for iter in 1:max_iter
        # Compute Euclidean gradient
        grad_eucl = grad_f(p)

        # Project to tangent space (Riemannian gradient)
        grad_riem = project_to_tangent_space(grad_eucl, p, ellipse)

        # Check convergence
        if norm(grad_riem) < tol
            if verbose
                println("Converged at iteration $iter")
            end
            break
        end

        # Retract: move along Riemannian gradient, stay on ellipse
        p_new = retract(p, -grad_riem, α, ellipse)

        # Check if still on ellipse (should be within numerical precision)
        x, y = p_new
        constraint_error = abs(x^2/ellipse.a^2 + y^2/ellipse.b^2 - 1.0)

        if verbose && iter % 100 == 0
            @printf("Iter %4d: f = %.6f, ||grad|| = %.6f, constraint_err = %.2e\n",
                    iter, f(p_new), norm(grad_riem), constraint_error)
        end

        p = p_new
        push!(trajectory, copy(p))
        push!(objectives, f(p))
    end

    if verbose
        println()
        @printf("Final point: [%.6f, %.6f]\n", p[1], p[2])
        @printf("Final objective: %.6f\n", f(p))

        # Verify constraint
        x, y = p
        constraint_val = x^2/ellipse.a^2 + y^2/ellipse.b^2
        @printf("Constraint x²/a² + y²/b² = %.10f (should be 1.0)\n", constraint_val)
    end

    return (
        optimum = p,
        trajectory = trajectory,
        objectives = objectives,
        converged = norm(project_to_tangent_space(grad_f(p), p, ellipse)) < tol
    )
end

"""
Visualize the optimization trajectory
"""
function plot_optimization(result, ellipse::Ellipse, f::Function, p0_original::Vector{Float64})
    # Create ellipse points for plotting
    θ = range(0, 2π, length=200)
    ellipse_x = ellipse.a .* cos.(θ)
    ellipse_y = ellipse.b .* sin.(θ)

    # Extract trajectory
    traj_x = [p[1] for p in result.trajectory]
    traj_y = [p[2] for p in result.trajectory]

    # Create contour plot of objective function
    x_range = range(-ellipse.a*1.5, ellipse.a*1.5, length=100)
    y_range = range(-ellipse.b*1.5, ellipse.b*1.5, length=100)
    Z = [f([x, y]) for y in y_range, x in x_range]

    # Plot
    p1 = contour(x_range, y_range, Z, levels=20,
                 xlabel="x", ylabel="y", title="Riemannian Optimization on Ellipse",
                 legend=:topright, colorbar=true)

    # Add ellipse
    plot!(p1, ellipse_x, ellipse_y, linewidth=3, color=:red, label="Ellipse constraint")

    # Add trajectory
    plot!(p1, traj_x, traj_y, linewidth=2, color=:blue,
          marker=:circle, markersize=3, label="Optimization path")

    # Mark original unprojected point (NOT on ellipse)
    scatter!(p1, [p0_original[1]], [p0_original[2]], markersize=10, color=:orange,
             marker=:xcross, label="Random initial (unprojected)", markerstrokewidth=3)

    # Draw arrow from unprojected to projected point
    plot!(p1, [p0_original[1], traj_x[1]], [p0_original[2], traj_y[1]],
          arrow=true, linewidth=2, linestyle=:dash, color=:orange,
          label="Projection to ellipse")

    # Mark start (projected) and end
    scatter!(p1, [traj_x[1]], [traj_y[1]], markersize=8, color=:green,
             marker=:star, label="Projected start (on ellipse)")
    scatter!(p1, [traj_x[end]], [traj_y[end]], markersize=8, color=:purple,
             marker=:star, label="Optimum")

    # Plot objective over iterations
    p2 = plot(result.objectives, xlabel="Iteration", ylabel="Objective value",
              title="Convergence", legend=false, linewidth=2, color=:blue)

    plot(p1, p2, layout=(1,2), size=(1200, 500))
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__

    # Define ellipse: x²/4 + y²/1 = 1  (a=2, b=1)
    ellipse = Ellipse(2.0, 1.0)
    println("Ellipse: x²/$(ellipse.a)² + y²/$(ellipse.b)² = 1")
    println()

    # Objective: minimize f(x,y) = x² + 2y²
    println("Objective: f(x,y) = x² + 2y²")
    println()

    # Test 1: Random initial point NOT on ellipse
    println("Test 1: Random initial point")
    println("-"^70)
    p0_random = [3.0, 2.0]  # NOT on ellipse
    result1 = riemannian_gradient_descent(
        objective_function,
        gradient_euclidean,
        ellipse,
        p0_random;
        max_iter=1000,
        α=0.1,
        verbose=true
    )

    p0_random2 = [-1.5, -2.5]  # NOT on ellipse
    result2 = riemannian_gradient_descent(
        objective_function,
        gradient_euclidean,
        ellipse,
        p0_random2;
        max_iter=1000,
        α=0.1,
        verbose=true
    )

  

    # Create visualization

    plt = plot_optimization(result1, ellipse, objective_function, p0_random)
    savefig(plt, "riemannian_ellipse_optimization.png")
end
