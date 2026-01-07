using LinearAlgebra
using Plots
using Printf

struct Ellipse
    a::Float64  # Semi-major axis
    b::Float64  # Semi-minor axis
end

function project_to_ellipse(p::Vector{Float64}, ellipse::Ellipse)
    # project_to_ellipse projects a point onto the ellipse by moving it along the line through the origin 
    #until it lies exactly on the ellipse.
    x, y = p
    a, b = ellipse.a, ellipse.b
    # Scale measures how much you need to shrink or stretch p to lie on the ellipse
    scale = sqrt((x^2)/(a^2) + (y^2)/(b^2))
    # Rescale point to lie on ellipse p^T*D*p = 1
    # where D = diagm([1/a^2, 1/b^2])
    return [x/scale, y/scale]
end

function tangent_vector(p::Vector{Float64}, ellipse::Ellipse)
    a, b = ellipse.a, ellipse.b
    x, y = p
    # Normal vector to ellipse at p
    # This vector points orthogonally outward from the ellipse
    # It defines the normal direction
    # p^T*D
    # normal = grad(F(p)), where F(p) = p^T D p
    normal = [2*x/a^2, 2*y/b^2]
    # Tangent vector is orthogonal to normal
    # Building a basis for the tangent space T_p*M
    # The condition for orthogonality is dot(tangent, normal) = 0 : p^T*D*v = 0
    tangent = [-normal[2], normal[1]]
    return tangent / norm(tangent)
end

function project_to_tangent_space(v::Vector{Float64}, p::Vector{Float64}, ellipse::Ellipse)
    # Project vector v onto the tangent space at point p on the ellipse
    t = tangent_vector(p, ellipse)
    # rebuilds the vector that lies entirely along the tangent direction
    # everything perpendicular to t is removed
    return (dot(v, t)) * t
    # This is the orthogonal projection onto the tangent space with respect to the Euclidean metric g(v,w)=v^Tw
end

function retract(p::Vector{Float64}, v::Vector{Float64}, α::Float64, ellipse::Ellipse)
    # Simple retraction: move along tangent vector and project back to ellipse
    p_new = p + α * v
    return project_to_ellipse(p_new, ellipse)
end



"""
Example objective function: minimize f(x,y) = x² + 2y²
Subject to constraint: x²/a² + y²/b² = 1
"""
function objective_function(p::Vector{Float64})
    x, y = p
    return x^2 + 2y^2
end

function gradient_euclidean(p::Vector{Float64})
    x, y = p
    return [2x, 4y]
end

function riemannian_gradient_descent(
    f::Function,
    grad_f::Function,
    ellipse::Ellipse;
    p0::Vector{Float64},
    α::Float64=0.1,
    max_iters::Int=1000,
    tol::Float64=1e-6,
    verbose::Bool=false
)

    # Project initial point to ellipse
    p = project_to_ellipse(p0, ellipse)
   
    trajectory = [copy(p)]
    objectives = [f(p)]

    for iter in 1:max_iters
        g_euc = grad_f(p)
        g_riem = project_to_tangent_space(g_euc, p, ellipse)

        p_new = retract(p, -g_riem, α, ellipse)

        push!(trajectory, copy(p_new))
        push!(objectives, f(p_new))

        if norm(p_new - p) < tol
            if verbose
                @printf("Converged in %d iterations.\n", iter)
            end
            break
        end

        p = p_new
    end
    
    return (optimum=p, trajectory=trajectory, objectives=objectives)
end