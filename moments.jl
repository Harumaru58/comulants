using XLSX, DataFrames, TensorDec, StatsBase, DynamicPolynomials

D = DataFrame(XLSX.readtable("data.xlsx", "ATN_sharp"))

groups = D[:, :Group] |> unique

data_by_groups = map(groups) do group 
    idx = findall(D[:, :Group] .== group)
    D[idx, 6:end] |> Matrix{Float64}
end
size(D, 1) == sum(size.(data_by_groups, 1))

decomps = map(data_by_groups) do X
    n = size(X, 1)

    # Mean-center
    mu = mean(X, dims=1)
    Xc = X .- mu

    @polyvar x[1:28]

    # 4th-order empirical moments
    iter =  Iterators.product(1:28, 1:28, 1:28, 1:28)
    M = zeros(28, 28, 28, 28)
    for (i,j,k,l) in iter
        if i <= j <= k <= l
            M[i,j,k,l] = sum(Xc[a,i] * Xc[a,j] * Xc[a,k] * Xc[a,l] for a in 1:n) / n
        end
    end

    for (i,j,k,l) in iter
        if i <= j <= k <= l && M[i,j,k,l] != 0
            perms = unique(Iterators.permutations([i,j,k,l]))
            for perm in perms
                M[perm...] = M[i,j,k,l]
            end
        end
    end
    
    T = sum(M[i,j,k,l] * x[i] * x[j] * x[k] * x[l] for (i,j,k,l) in iter)
    F = TensorDec.approximate(T, 5)
    F
end


