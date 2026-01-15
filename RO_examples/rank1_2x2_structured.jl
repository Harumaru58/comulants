# Rank-1 2×2 matrix optimization using structured Riemannian optimization
# Uses explicit Rank1 type with tangent space projection

using LinearAlgebra
using Random

# Rank-1 matrix representation: X = σ * u * v'
struct Rank1
    σ::Float64
    u::Vector{Float64}
    v::Vector{Float64}
end

# Convert Rank1 to matrix
matrix(X::Rank1) = X.σ * X.u * X.v'

# Tangent projection: map ambient gradient G to tangent space at X
function tangent_projection(G, X::Rank1)
    u, v, σ = X.u, X.v, X.σ

    α = dot(u, G * v)
    wu = (G * v) / σ - α * u / σ
    wv = (G' * u) / σ - α * v / σ

    # Enforce orthogonality numerically
    wu -= dot(wu, u) * u
    wv -= dot(wv, v) * v

    return α, wu, wv
end

# Retraction: move in tangent direction and return to manifold
function retract(X::Rank1, α, wu, wv, stepsize)
    # Update in ambient space
    M = matrix(X) - stepsize * (α * X.u * X.v' + X.σ * wu * X.v' + X.σ * X.u * wv')
    
    # Project back to rank-1 via SVD
    U, S, V = svd(M)
    
    # Ensure unit vectors
    u_new = U[:, 1]
    v_new = V[:, 1]
    σ_new = S[1]
    
    return Rank1(σ_new, u_new, v_new)
end

# Target matrix
const TARGET = [3.0 1.2;
                2.0 0.5]

# Cost function
cost(X::Rank1) = 0.5 * norm(matrix(X) - TARGET)^2

# Euclidean gradient
egrad(X::Rank1) = matrix(X) - TARGET

# Riemannian gradient descent
function optimize_rank1(; stepsize=0.2, maxiter=500, tol=1e-8, seed=42)
    Random.seed!(seed)
    
    # Initialize random rank-1 matrix
    M0 = randn(2, 2)
    U, S, V = svd(M0)
    X = Rank1(S[1], U[:, 1], V[:, 1])
    
    for k in 1:maxiter
        # Compute Euclidean gradient
        G = egrad(X)
        
        # Project to tangent space
        α, wu, wv = tangent_projection(G, X)
        
        # Riemannian gradient norm
        gnorm = sqrt(α^2 + X.σ^2 * (norm(wu)^2 + norm(wv)^2))
        
        # Check convergence
        if gnorm < tol
            return X, k, gnorm
        end
        
        # Retract to manifold
        X = retract(X, α, wu, wv, stepsize)
    end
    
    gnorm = let G = egrad(X), (α, wu, wv) = tangent_projection(G, X)
        sqrt(α^2 + X.σ^2 * (norm(wu)^2 + norm(wv)^2))
    end
    
    return X, maxiter, gnorm
end

# Run optimization
X_opt, iters, grad_norm = optimize_rank1()

# Display results
println("="^60)
println("Rank-1 2×2 Matrix Optimization (Structured Representation)")
println("="^60)
println("\nConverged in $iters iterations")
println("Final gradient norm: $grad_norm")
println("Final cost: $(cost(X_opt))")

println("\n" * "-"^60)
println("Solution: X = σ × u ⊗ v")
println("-"^60)
println("σ = $(round(X_opt.σ, digits=4))")
println("u = $(round.(X_opt.u, digits=4))")
println("v = $(round.(X_opt.v, digits=4))")

println("\n" * "-"^60)
println("Matrix form:")
println("-"^60)
display(round.(matrix(X_opt), digits=4))

println("\n\n" * "-"^60)
println("Target:")
println("-"^60)
display(TARGET)

println("\n\n" * "-"^60)
println("Error:")
println("-"^60)
display(round.(matrix(X_opt) - TARGET, digits=4))

println("\n\n" * "="^60)
println("Verification:")
println("="^60)
println("Orthonormality: ||u|| = $(norm(X_opt.u)), ||v|| = $(norm(X_opt.v))")
println("Reconstruction: ||X - σuv'|| = $(norm(matrix(X_opt) - X_opt.σ * X_opt.u * X_opt.v'))")
println("Target rank: $(rank(TARGET))")
println("="^60)
