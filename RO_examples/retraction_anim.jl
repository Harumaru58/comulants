using LinearAlgebra
using Plots

# Ellipse parameters
struct Ellipse
    a::Float64
    b::Float64
end

# Angle-based projection
function project_to_ellipse_angle(p::Vector{Float64}, ellipse::Ellipse)
    x, y = p
    a, b = ellipse.a, ellipse.b
    θ = atan(b*y, a*x)
    return [a * cos(θ), b * sin(θ)]
end

# Retraction
function retract_angle(p::Vector{Float64}, v::Vector{Float64}, α::Float64, ellipse::Ellipse)
    p_new = p + α * v
    return project_to_ellipse_angle(p_new, ellipse)
end

# Tangent vector
function tangent_vector(p::Vector{Float64}, ellipse::Ellipse)
    a, b = ellipse.a, ellipse.b
    normal = [2*p[1]/a^2, 2*p[2]/b^2]
    tangent = [-normal[2], normal[1]]
    return tangent / norm(tangent)
end


# -----------------------
# Numerical check
# -----------------------
function check_retraction()
    ellipse = Ellipse(3.0, 2.0)
    p = [3.0, 0.0]  # point on ellipse
    v = tangent_vector(p, ellipse)  # tangent vector at p
    t = 1e-8  # small step

    # Finite difference approximation of derivative
    DRv = (retract_angle(p, t*v, ellipse) - p) / t

    println("Tangent vector v:       ", v)
    println("Approximated DR_p(0)v: ", DRv)
    println("Difference:             ", DRv - v)
end

check_retraction()
# -----------------------
# Setup
# -----------------------
ellipse = Ellipse(3.0, 2.0)
p = [3.0, 0.0]                # point on ellipse
v = tangent_vector(p, ellipse)  # tangent vector

# Tangent line
t_line = LinRange(-1.0, 1.0, 100)
tangent_line = [p .+ s*v for s in t_line]

# Ellipse points
θ = LinRange(-π, π, 500)
ellipse_points = [ellipse.a*cos.(θ) ellipse.b*sin.(θ)]

# Animation: t goes from small to larger
anim = @animate for t in LinRange(0.0, 2.0, 25)
    # Step and retraction
    p_step = p + t*v
    p_retract = retract_angle(p, t*v, 1.0, ellipse)

    plot(ellipse_points[:,1], ellipse_points[:,2], lw=2, label="Ellipse", aspect_ratio=:equal, legend=:topleft)
    scatter!([p[1]], [p[2]], color=:red, ms=6, label="Point p")
    plot!([pt[1] for pt in tangent_line], [pt[2] for pt in tangent_line], linestyle=:dash, color=:green, label="Tangent line")
    quiver!([p[1]], [p[2]], quiver=([v[1]*0.5],[v[2]*0.5]), color=:green, label="Tangent vector v")
    scatter!([p_step[1]], [p_step[2]], color=:orange, ms=5, label="p + t v (Euclidean step)")
    scatter!([p_retract[1]], [p_retract[2]], color=:blue, ms=5, label="Retracted R_p(t v)")
    plot!([p_step[1], p_retract[1]], [p_step[2], p_retract[2]], color=:black, linestyle=:dot, label="O(t²) deviation")
    xlabel!("x")
    ylabel!("y")
    title!("t = $(round(t, digits=2)): DR_p(0)v ~ tangent vector")
end

# Save the animation as a gif
gif(anim, "ellipse_retraction.gif", fps=5)
