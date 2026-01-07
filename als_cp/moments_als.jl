using XLSX, DataFrames, StatsBase, Combinatorics
include("symmetric_als.jl")

# Read data
D = DataFrame(XLSX.readtable("data.xlsx", "ATN_sharp"))

groups = D[:, :Group] |> unique

data_by_groups = map(groups) do group 
    idx = findall(D[:, :Group] .== group)
    D[idx, 6:end] |> Matrix{Float64}
end


# Process each group
decomps = map(enumerate(data_by_groups)) do (group_idx, X)
    group_name = groups[group_idx]
    
    n = size(X, 1)
    n_features = size(X, 2)
    n_modes = 4  # 4th-order tensor

    
    # Mean-center
    mu = mean(X, dims=1)
    Xc = X .- mu
    
    # 4th-order empirical moments
    iter = Iterators.product(1:n_features, 1:n_features, 1:n_features, 1:n_features)
    M = zeros(n_features, n_features, n_features, n_features)
    
    for (i,j,k,l) in iter
        if i <= j <= k <= l
            M[i,j,k,l] = sum(Xc[a,i] * Xc[a,j] * Xc[a,k] * Xc[a,l] for a in 1:n) / n
        end
    end
    
    # Fill in symmetric entries
    for (i,j,k,l) in iter
        if i <= j <= k <= l && M[i,j,k,l] != 0
            perms = unique(permutations([i,j,k,l]))
            for perm in perms
                M[perm...] = M[i,j,k,l]
            end
        end
    end
    
    # Use symmetric ALS CP decomposition (all factors equal)
    result = symmetric_als_cp(M, 5; max_iter=1000, tol=1e-6, verbose=false)
    
    # Convert to same format as regular ALS (vector of factors)
    # For symmetric decomposition, all factors are the same
    factors_all = fill(result.factor, n_modes)
    
    println("  Relative error: $(round(result.error, digits=4))")
    println("  Iterations: $(result.n_iter)")
    println("  Converged: $(result.converged)")
    println()
    
    # Return decomposition results (compatible format)
    Dict(
        :group => group_name,
        :factors => factors_all,
        :weights => hasfield(typeof(result), :weights) ? result.weights : ones(5),
        :relative_error => result.error,
        :n_iter => result.n_iter,
        :converged => result.converged,
        :moment_tensor => M,
        :symmetric_factor => result.factor  # Store the single symmetric factor
    )
end

for (idx, decomp) in enumerate(decomps)
    println("Group $(idx): $(decomp[:group])")
    println("  Error: $(round(decomp[:relative_error], digits=4))")
    println("  Iterations: $(decomp[:n_iter])")
    println("  Converged: $(decomp[:converged])")
    println("  Component weights: $(round.(decomp[:weights], digits=2))")
    println()
end

