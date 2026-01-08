module MomentTensorConstruction

using LinearAlgebra
using Statistics
using XLSX
using DataFrames

export construct_third_moment_tensor

"""
    construct_third_moment_tensor(filepath; n_features=28)

Reads CSF biomarker data from an Excel file and constructs a partially symmetric
3rd-order moment tensor stacked over clinical groups.

# Returns
- `T::Array{Float64,4}`: Tensor of size (28, 28, 28, 4) 
  - First 3 dimensions: symmetric 3rd-order moment
  - 4th dimension: groups (CU_A-T-, AD_A+T+, CBS_A-T+, CBS-AD_A+T+)
- `groups::Vector`: Vector of unique group labels

"""
function construct_third_moment_tensor(filepath::AbstractString;
                                      n_features::Int=28)
    # Read Excel file
    data = XLSX.readxlsx(filepath)[1] |> XLSX.gettable |> DataFrame
    
    # Extract group labels from column 1
    group_labels = Vector(data[:, 1])
    
    # Extract 28 biomarker features from columns 6-33 (APOE_total through C1QC)
    X = Matrix(data[:, 6:33])
    
    # Verify dimensions
    @assert size(X, 2) == n_features "Expected $n_features features, got $(size(X, 2))"
    
    groups = unique(group_labels)
    G = length(groups)
    
    println("Constructing tensor for $G groups:")
    for (idx, g) in enumerate(groups)
        n_samples = sum(group_labels .== g)
        println("  [$idx] $g: $n_samples samples")
    end
    
    # Initialize 4D tensor: (28, 28, 28, G)
    T = zeros(Float64, n_features, n_features, n_features, G)
    
    # Compute 3rd-order moment for each group
    for (gidx, g) in enumerate(groups)
        idx = findall(group_labels .== g)
        Xg = X[idx, :]
        
        # Center the data (remove mean)
        μ = mean(Xg, dims=1)
        Xc = Xg .- μ
        Ng = size(Xc, 1)
        
        # Compute symmetric 3rd-order moment
        for n in 1:Ng
            x = view(Xc, n, :)
            # Only compute upper triangular entries, then symmetrize
            for i in 1:n_features, j in i:n_features, k in j:n_features
                val = x[i] * x[j] * x[k]
                # Fill all 6 permutations for symmetric tensor
                T[i,j,k,gidx] += val
                T[i,k,j,gidx] += val
                T[j,i,k,gidx] += val
                T[j,k,i,gidx] += val
                T[k,i,j,gidx] += val
                T[k,j,i,gidx] += val
            end
        end
        
        # Normalize by sample size
        T[:,:,:,gidx] ./= Ng
    end
    
    return T, groups
end

end 

# Execute the function
using .MomentTensorConstruction

T, groups = construct_third_moment_tensor("data/data.xlsx")
println("\nGroups found: ", groups)
println("Tensor shape: ", size(T))
