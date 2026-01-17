using LinearAlgebra
using Random
using Manifolds
using Manopt
using RecursiveArrayTools

m, n = 30, 20
Random.seed!(42)
A = randn(m, n)

const run_no_normalization = true
const run_simple_normalization = true

M = ProductManifold(Euclidean(1), Sphere(m - 1), Sphere(n - 1))

rank1(p) = (p.x[1][1]) * p.x[2] * p.x[3]'
retract_rank1(Y) = begin
    U, S, V = svd(Y)
    U[:, 1] * S[1] * V[:, 1]'
end
project_to_tangent(X, G) = begin
    U, _, V = svd(X)
    u = U[:, 1]
    v = V[:, 1]
    P_u = u * u'
    P_v = v * v'
    P_u * G + G * P_v - P_u * G * P_v
end

function run_manopt()
    p0 = ArrayPartition([1.0], normalize(randn(m)), normalize(randn(n)))
    cost = (M, p) -> norm(A - rank1(p))^2
    egrad = (M, p) -> begin
        λ = p.x[1][1]
        u = p.x[2]
        v = p.x[3]
        G = -2 * (A - λ * u * v')
        ArrayPartition([sum(G .* (u * v'))], λ * (G * v), λ * (G' * u))
    end
    gradient_descent(
        M, cost, egrad, p0;
        stepsize=Manopt.ConstantStepsize(M, 0.1),
        stopping_criterion=StopWhenAny(
            StopAfterIteration(500),
            StopWhenGradientNormLess(1e-8)
        )
    )
end

function run_svd_retraction()
    X = retract_rank1(randn(m, n))
    gnorm = Inf
    for _ in 1:500
        Rgrad = project_to_tangent(X, X - A)
        gnorm = norm(Rgrad)
        gnorm < 1e-8 && break
        X = retract_rank1(X - 0.2 * Rgrad)
    end
    return X, gnorm
end

function run_no_norm()
    λ = 1.0
    u = randn(m)
    v = randn(n)
    gnorm = Inf
    for _ in 1:2000
        X = λ * u * v'
        G = -2 * (A - X)
        dλ = sum(G .* (u * v'))
        du = λ * (G * v)
        dv = λ * (G' * u)
        gnorm = sqrt(dλ^2 + norm(du)^2 + norm(dv)^2)
        gnorm < 1e-8 && break
        λ -= 0.01 * dλ
        u -= 0.01 * du
        v -= 0.01 * dv

        # Absorb scaling into λ to keep u, v well-conditioned
        su = norm(u)
        sv = norm(v)
        if su > 0 && sv > 0
            λ *= su * sv
            u /= su
            v /= sv
        end
    end
    return λ, u, v, gnorm
end

function run_simple_norm()
    λ = 1.0
    u = normalize(randn(m))
    v = normalize(randn(n))
    gnorm = Inf
    for _ in 1:2000
        X = λ * u * v'
        G = -2 * (A - X)
        dλ = sum(G .* (u * v'))
        du = λ * (G * v)
        dv = λ * (G' * u)
        gnorm = sqrt(dλ^2 + norm(du)^2 + norm(dv)^2)
        gnorm < 1e-8 && break
        λ -= 0.01 * dλ
        u = normalize(u - 0.01 * du)
        v = normalize(v - 0.01 * dv)
    end
    return λ, u, v, gnorm
end

p = run_manopt()
λp = p.x[1][1]; up = p.x[2]; vp = p.x[3]
Xp = λp * up * vp'

Xr, gnr = run_svd_retraction()

λn = 0.0; un = zeros(m); vn = zeros(n); gnn = NaN
if run_no_normalization
    λn, un, vn, gnn = run_no_norm()
end

λs = 0.0; us = zeros(m); vs = zeros(n); gns = NaN
if run_simple_normalization
    λs, us, vs, gns = run_simple_norm()
end

U, S, V = svd(A)
Xbest = S[1] * U[:, 1] * V[:, 1]'
println("product err: ", norm(A - Xp))
println("svd retract err: ", norm(A - Xr), " (grad ", gnr, ")")
if run_no_normalization
    println("no norm err: ", norm(A - λn * un * vn'), " (grad ", gnn, ")")
end
if run_simple_normalization
    println("simple norm err: ", norm(A - λs * us * vs'), " (grad ", gns, ")")
end
println("svd best err: ", norm(A - Xbest))