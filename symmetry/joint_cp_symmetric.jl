"""
Joint CP Decomposition with Symmetric Structure Interpretation

This script performs:
1. Load 4D moment tensor (28×28×28×4) from biomarker data
2. Joint CP decomposition across all groups
3. Symmetrize factor matrices (exploit symmetric structure)
4. Compute group-specific residuals
5. Optional residual CP for group-specific patterns
6. Report reconstruction errors and interpretations

Dependencies:
  using Pkg
  Pkg.add(["XLSX", "DataFrames", "TensorDecompositions"])
"""

using LinearAlgebra
using Statistics
using XLSX
using DataFrames
using TensorDecompositions

##############################################################################
# Data Loading
##############################################################################

"""
Load biomarker data and construct 4D moment tensor (28×28×28×G)
"""
function load_moment_tensor(filepath::AbstractString; n_features::Int=28)
    println("="^70)
    println("Loading data from: $filepath")
    println("="^70)
    
    # Read Excel file
    data = XLSX.readxlsx(filepath)[1] |> XLSX.gettable |> DataFrame
    
    # Extract group labels from column 1
    group_labels = Vector(data[:, 1])
    
    # Extract 28 biomarker features from columns 6-33
    X = Matrix(data[:, 6:33])
    
    # Handle missing values (replace with column mean)
    for j in 1:size(X, 2)
        col = X[:, j]
        missing_idx = ismissing.(col)
        if any(missing_idx)
            col_mean = mean(skipmissing(col))
            X[missing_idx, j] .= col_mean
        end
    end
    
    # Convert to Float64
    X = Float64.(X)
    
    @assert size(X, 2) == n_features "Expected $n_features features, got $(size(X, 2))"
    
    groups = unique(group_labels)
    G = length(groups)
    
    println("\nGroups found: $G")
    for (idx, g) in enumerate(groups)
        n_samples = sum(group_labels .== g)
        println("  [$idx] $g: $n_samples samples")
    end
    
    # Initialize 4D tensor: (28, 28, 28, G)
    T = zeros(Float64, n_features, n_features, n_features, G)
    
    # Compute 3rd-order moment for each group
    println("\nComputing 3rd-order moment tensors...")
    for (gidx, g) in enumerate(groups)
        idx = findall(group_labels .== g)
        Xg = X[idx, :]
        
        # Center the data
        μ = mean(Xg, dims=1)
        Xc = Xg .- μ
        Ng = size(Xc, 1)
        
        # Compute symmetric 3rd-order moment
        for n in 1:Ng
            x = view(Xc, n, :)
            # Only compute upper triangular entries, then symmetrize
            for i in 1:n_features, j in i:n_features, k in j:n_features
                val = x[i] * x[j] * x[k]
                # Fill all 6 permutations for full symmetry
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
        
        println("  Group $gidx: tensor norm = $(norm(T[:,:,:,gidx]))")
    end
    
    return T, groups
end

##############################################################################
# Utility Functions
##############################################################################

"""
Symmetrize three factor matrices into one symmetric mode matrix.
Averages A, B, C and normalizes each column.
"""
function symmetrize_abc(A, B, C)
    n, r = size(A)
    S = zeros(n, r)
    for k in 1:r
        s = (A[:,k] + B[:,k] + C[:,k]) / 3
        s_norm = norm(s)
        if s_norm > 1e-10
            S[:,k] = s / s_norm
        else
            S[:,k] = s  # Keep zero if all near zero
        end
    end
    return S
end

"""
Reconstruct a symmetric CP tensor: T̂ = Σᵣ λᵣ sᵣ⊗sᵣ⊗sᵣ
"""
function reconstruct_symmetric(S, λ)
    n, r = size(S)
    T̂ = zeros(n, n, n)
    for k in 1:r
        T̂ .+= λ[k] .* reshape(S[:,k], :,1,1) .*
                       reshape(S[:,k], 1,:,1) .*
                       reshape(S[:,k], 1,1,:)
    end
    return T̂
end

"""
Compute residual tensor after removing symmetric CP approximation
"""
function compute_residual(Xg, S, λ)
    R = copy(Xg)
    n, r = size(S)
    for k in 1:r
        R .-= λ[k] .* reshape(S[:,k], :,1,1) .*
                       reshape(S[:,k], 1,:,1) .*
                       reshape(S[:,k], 1,1,:)
    end
    return R
end

##############################################################################
# Main Analysis Pipeline
##############################################################################

function joint_symmetric_cp_analysis(
    filepath::AbstractString="data/data.xlsx";
    joint_rank::Int=5,
    residual_rank::Int=1,
    residual_threshold::Float64=0.15,
    maxiter::Int=800,
    tol::Float64=1e-8
)
    println("\n" * "="^70)
    println("JOINT CP DECOMPOSITION WITH SYMMETRIC STRUCTURE")
    println("="^70)
    println()
    
    # 1. Load data and construct 4D moment tensor
    X, group_names = load_moment_tensor(filepath)
    n_groups = length(group_names)
    
    @assert ndims(X) == 4 "X must be 4D array (28×28×28×G)"
    println("\nTensor shape: $(size(X))")
    println()
    
    # 2. Joint CP decomposition
    println("="^70)
    println("STEP 1: Joint CP Decomposition (rank=$joint_rank)")
    println("="^70)
    println()
    
    cp_joint = cp_als(X, joint_rank; maxiter=maxiter, tol=tol)
    
    # Extract factors: A, B, C ∈ ℝ^{28×r}, G ∈ ℝ^{n_groups×r}
    A, B, C, G = cp_joint.factors
    λ = cp_joint.lambdas
    
    println("CP decomposition completed.")
    println("  Factors: A, B, C ∈ ℝ^($(size(A, 1))×$(size(A, 2)))")
    println("  Group loadings G ∈ ℝ^($(size(G, 1))×$(size(G, 2)))")
    println("  Component weights λ: ", λ)
    println()
    
    # 3. Symmetrize joint factors
    println("="^70)
    println("STEP 2: Symmetrize Factor Matrices")
    println("="^70)
    println()
    
    S_joint = symmetrize_abc(A, B, C)
    
    println("Symmetrized factors S ∈ ℝ^(28×$joint_rank)")
    println("  Column norms: ", [norm(S_joint[:,k]) for k in 1:joint_rank])
    println()
    
    # 4. Compute residuals per group
    println("="^70)
    println("STEP 3: Compute Group-Specific Residuals")
    println("="^70)
    println()
    
    Residuals = []
    for g in 1:n_groups
        # Group-specific weights
        λ_g = G[g, :] .* λ
        
        R_g = compute_residual(X[:,:,:,g], S_joint, λ_g)
        push!(Residuals, R_g)
        
        rel_norm = norm(R_g) / norm(X[:,:,:,g])
        println("Group $g ($(group_names[g])):")
        println("  Original norm: $(norm(X[:,:,:,g]))")
        println("  Residual norm: $(norm(R_g))")
        println("  Relative residual: $(rel_norm)")
        println()
    end
    
    # 5. Residual CP per group (if residual is significant)
    println("="^70)
    println("STEP 4: Residual CP Decomposition (threshold=$residual_threshold)")
    println("="^70)
    println()
    
    res_cp = Vector{Union{Nothing, Any}}(undef, n_groups)
    residual_modes = Vector{Union{Nothing, Tuple{Matrix{Float64}, Vector{Float64}}}}(undef, n_groups)
    
    for g in 1:n_groups
        R_g = Residuals[g]
        rel_norm = norm(R_g) / norm(X[:,:,:,g])
        
        if rel_norm > residual_threshold
            println("Group $g ($(group_names[g])): residual significant, running CP...")
            
            # Run CP on residual
            res_cp[g] = cp_als(R_g, residual_rank; maxiter=500, tol=tol)
            
            # Symmetrize residual factors
            A_r, B_r, C_r = res_cp[g].factors
            S_r = symmetrize_abc(A_r, B_r, C_r)
            λ_r = res_cp[g].lambdas
            
            residual_modes[g] = (S_r, λ_r)
            println("  Residual CP completed, symmetrized.")
            println()
        else
            println("Group $g ($(group_names[g])): residual small, skipped.")
            res_cp[g] = nothing
            residual_modes[g] = nothing
            println()
        end
    end
    
    # 6. Reconstruction errors
    println("="^70)
    println("STEP 5: Reconstruction Error Report")
    println("="^70)
    println()
    
    for g in 1:n_groups
        T_g = X[:,:,:,g]
        λ_g = G[g, :] .* λ
        T̂_joint = reconstruct_symmetric(S_joint, λ_g)
        
        err_joint = norm(T_g - T̂_joint) / norm(T_g)
        
        println("Group $g ($(group_names[g])):")
        println("  Joint CP error: $(err_joint)")
        
        if residual_modes[g] !== nothing
            (S_r, λ_r) = residual_modes[g]
            T̂_res = reconstruct_symmetric(S_r, λ_r)
            T̂_total = T̂_joint + T̂_res
            
            err_residual = norm(Residuals[g] - T̂_res) / norm(Residuals[g])
            err_total = norm(T_g - T̂_total) / norm(T_g)
            
            println("  Residual CP error: $(err_residual)")
            println("  Total reconstruction error: $(err_total)")
        end
        println()
    end
    
    # 7. Interpretation
    println("="^70)
    println("INTERPRETATION: Symmetric Structure")
    println("="^70)
    println()
    
    println("Joint symmetric components (shared across groups):")
    for k in 1:joint_rank
        top_idx = sortperm(abs.(S_joint[:,k]), rev=true)[1:5]
        println("  Component $k (weight=$(λ[k])):")
        println("    Top 5 biomarkers: $top_idx")
        println("    Loadings: ", S_joint[top_idx, k])
    end
    println()
    
    println("Group-specific loadings on joint components:")
    for g in 1:n_groups
        println("  $(group_names[g]): ", G[g, :])
    end
    println()
    
    return (
        tensor = X,
        groups = group_names,
        joint_factors = S_joint,
        joint_weights = λ,
        group_loadings = G,
        residuals = Residuals,
        residual_modes = residual_modes
    )
end

##############################################################################
# Execute Analysis
##############################################################################

if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting joint symmetric CP analysis...")
    println()
    
    results = joint_symmetric_cp_analysis(
        "data/data.xlsx";
        joint_rank=5,
        residual_rank=1,
        residual_threshold=0.15,
        maxiter=800,
        tol=1e-8
    )
    
    println("="^70)
    println("ANALYSIS COMPLETED")
    println("="^70)
    println()
    println("Results available in 'results' variable:")
    println("  - results.tensor: 4D moment tensor")
    println("  - results.joint_factors: Symmetrized factor matrix (28×5)")
    println("  - results.joint_weights: Component weights")
    println("  - results.group_loadings: Group-specific loadings on components")
    println("  - results.residuals: Group-specific residual tensors")
    println("  - results.residual_modes: Optional group-specific components")
    println()
    println("Biological interpretation:")
    println("  • Joint components = shared biomarker interaction patterns")
    println("  • Group loadings = how each clinical group uses these patterns")
    println("  • Residuals = group-specific deviations from shared structure")
end
