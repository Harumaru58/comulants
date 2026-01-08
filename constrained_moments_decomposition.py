"""
Constrained Moment Tensor Decomposition using Riemannian Optimization

Integrates:
1. Load biomarker data from data.xlsx
2. Compute 3rd-order moment tensors per group
3. Compute empirical covariance per group
4. Decompose with ellipsoidal constraints via Julia
"""

import numpy as np
import pandas as pd
from itertools import permutations
import subprocess
import json
import tempfile
import os


def load_biomarker_data(filepath="data/data.xlsx", sheet_name="ATN_sharp"):
    """Load biomarker data from Excel file."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # First 5 columns are metadata
    biomarker_cols = df.columns[5:].tolist()
    groups = df['Group'].unique()

    data_by_groups = {}
    for group in groups:
        idx = df['Group'] == group
        group_data = df.loc[idx, biomarker_cols].fillna(df[biomarker_cols].mean()).astype(float)
        data_by_groups[group] = group_data.values

    return data_by_groups, biomarker_cols


def compute_empirical_covariance(X):
    """Compute empirical covariance matrix."""
    n, n_features = X.shape
    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu

    # Covariance: Σ = (1/n) Xᵀ X
    Sigma = (Xc.T @ Xc) / n

    # Add small regularization for numerical stability
    Sigma = Sigma + 1e-6 * np.eye(n_features)

    return Sigma


def compute_3rd_order_moments(X):
    """Compute 3rd-order empirical moments tensor (symmetric)."""
    n, n_features = X.shape
    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu

    M = np.zeros((n_features, n_features, n_features))

    for i in range(n_features):
        for j in range(i, n_features):
            for k in range(j, n_features):
                moment = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k])
                # Fill all permutations for symmetry
                for idx in set(permutations([i, j, k])):
                    M[idx] = moment

    return M


def save_for_julia(moment_tensor, covariance, rank, output_prefix):
    """Save data in format Julia can load."""
    # Save as numpy arrays
    np.save(f"{output_prefix}_moment.npy", moment_tensor)
    np.save(f"{output_prefix}_covariance.npy", covariance)

    # Save metadata
    metadata = {
        'rank': rank,
        'shape': moment_tensor.shape,
        'n_biomarkers': covariance.shape[0]
    }
    with open(f"{output_prefix}_meta.json", 'w') as f:
        json.dump(metadata, f)

    return f"{output_prefix}_moment.npy", f"{output_prefix}_covariance.npy"


def run_constrained_decomposition_julia(moment_path, cov_path, rank,
                                       confidence_level=2.0, verbose=True):
    """
    Call Julia script to run constrained CP decomposition.

    This creates a Julia script that:
    1. Loads the moment tensor and covariance
    2. Creates ellipsoid from covariance
    3. Runs constrained ALS
    4. Saves results back to Python
    """

    # Get current directory for includes
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create temporary Julia script
    julia_script = f"""
using NPZ
using JSON
using LinearAlgebra
include("{current_dir}/riemannian_ellipsoid.jl")
include("{current_dir}/constrained_cp_decomposition.jl")

# Load data
M = npzread("{moment_path}")
Sigma = npzread("{cov_path}")

rank = {rank}
confidence_level = {confidence_level}

println("Loaded moment tensor: ", size(M))
println("Loaded covariance: ", size(Sigma))
println()

# Create ellipsoid from covariance
ellipsoid = CovarianceEllipsoid(Sigma, confidence_level)

# For symmetric 3rd-order tensor, all modes use same ellipsoid
ellipsoids = [ellipsoid, ellipsoid, ellipsoid]

# Run constrained decomposition
result = constrained_als_cp(M, rank, ellipsoids; max_iter=500, tol=1e-6, verbose=true)

# Save results
output_prefix = "{moment_path[:-4]}"
npzwrite(output_prefix * "_factors.npz",
         Dict("factor1" => result.factors[1],
              "factor2" => result.factors[2],
              "factor3" => result.factors[3],
              "weights" => result.weights,
              "error" => result.error))

println()
println("Saved results to: ", output_prefix * "_factors.npz")
"""

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
        f.write(julia_script)
        script_path = f.name

    try:
        # Run Julia script
        if verbose:
            print("Running constrained decomposition in Julia...")
            print()

        result = subprocess.run(['julia', script_path],
                              capture_output=True,
                              text=True,
                              timeout=300)

        if verbose:
            print(result.stdout)

        if result.returncode != 0:
            print("Julia error:")
            print(result.stderr)
            return None

        # Load results
        result_path = moment_path[:-4] + "_factors.npz"
        if os.path.exists(result_path):
            results = np.load(result_path)
            return {
                'factors': [results['factor1'], results['factor2'], results['factor3']],
                'weights': results['weights'],
                'relative_error': float(results['error'])
            }
        else:
            print(f"Results file not found: {result_path}")
            return None

    finally:
        # Cleanup
        if os.path.exists(script_path):
            os.remove(script_path)


def constrained_moments_decomposition(
    filepath="data/data.xlsx",
    sheet_name="ATN_sharp",
    rank=5,
    confidence_level=2.0,
    use_julia=True,
    verbose=True
):
    """
    Main function: Load data, compute moments, decompose with constraints.

    Parameters:
    -----------
    filepath : str
        Path to Excel data file
    sheet_name : str
        Sheet name in Excel
    rank : int
        CP decomposition rank
    confidence_level : float
        Confidence level for ellipsoid (2.0 = 2-sigma = ~95%)
    use_julia : bool
        If True, use Julia for constrained decomposition (more efficient)
        If False, use Python approximation
    verbose : bool
        Print progress
    """

    if verbose:
        print("="*70)
        print("Constrained Moment Tensor Decomposition")
        print("="*70)
        print()

    # Load data
    if verbose:
        print(f"Loading data from {filepath}...")

    data_by_groups, biomarker_names = load_biomarker_data(filepath, sheet_name)

    if verbose:
        print(f"Found {len(data_by_groups)} groups")
        print(f"Found {len(biomarker_names)} biomarkers")
        print()

    results = {}

    for group_name, X in data_by_groups.items():
        if verbose:
            print(f"Processing group: {group_name}")
            print(f"  Samples: {X.shape[0]}")

        # Compute moment tensor
        M = compute_3rd_order_moments(X)

        # Compute covariance
        Sigma = compute_empirical_covariance(X)

        if verbose:
            print(f"  Moment tensor shape: {M.shape}")
            print(f"  Covariance shape: {Sigma.shape}")
            print(f"  Covariance condition number: {np.linalg.cond(Sigma):.2f}")

        if use_julia:
            # Use Julia for constrained optimization
            with tempfile.TemporaryDirectory() as tmpdir:
                output_prefix = os.path.join(tmpdir, f"group_{group_name}")
                moment_path, cov_path = save_for_julia(M, Sigma, rank, output_prefix)

                result = run_constrained_decomposition_julia(
                    moment_path, cov_path, rank,
                    confidence_level=confidence_level,
                    verbose=verbose
                )

                if result:
                    result['group'] = group_name
                    result['moment_tensor'] = M
                    result['covariance'] = Sigma
                    results[group_name] = result
        else:
            # Python fallback (unconstrained - for comparison)
            print("  Note: Python version runs unconstrained decomposition")
            print("  Use use_julia=True for constrained version")

            from symmetry.moments_3rd_order import decompose_tensor
            decomp = decompose_tensor(M, rank=rank)
            decomp['group'] = group_name
            decomp['moment_tensor'] = M
            decomp['covariance'] = Sigma
            results[group_name] = decomp

        if verbose:
            print()

    if verbose:
        print("="*70)
        print("Summary")
        print("="*70)

        for group_name, result in results.items():
            if result:
                print(f"{group_name}: error = {result['relative_error']:.4f}")

    return results, biomarker_names


def compare_constrained_vs_unconstrained(
    filepath="data.xlsx",
    rank=5,
    confidence_level=2.0
):
    """
    Compare constrained vs unconstrained decomposition results.
    """
    print("Running CONSTRAINED decomposition...")
    print()

    results_constrained, biomarker_names = constrained_moments_decomposition(
        filepath=filepath,
        rank=rank,
        confidence_level=confidence_level,
        use_julia=True,
        verbose=True
    )

    print("\n" + "="*70)
    print("\nRunning UNCONSTRAINED decomposition...")
    print()

    from symmetry.moments_3rd_order import main as moments_3rd_main
    results_unconstrained = moments_3rd_main(filepath=filepath, rank=rank)

    # Compare
    print("\n" + "="*70)
    print("Comparison: Constrained vs Unconstrained")
    print("="*70)
    print()

    for i, decomp in enumerate(results_unconstrained):
        group = decomp['group']
        error_uncon = decomp['relative_error']

        if group in results_constrained:
            error_con = results_constrained[group]['relative_error']
            print(f"{group}:")
            print(f"  Unconstrained error: {error_uncon:.4f}")
            print(f"  Constrained error:   {error_con:.4f}")
            print(f"  Difference:          {error_con - error_uncon:+.4f}")
            print()

    return results_constrained, results_unconstrained


if __name__ == "__main__":
    # Check if Julia and required packages are available
    try:
        result = subprocess.run(['julia', '--version'],
                              capture_output=True,
                              timeout=5)
        print("Julia found:", result.stdout.decode())
        print()

        # Check for NPZ package
        check_script = "using NPZ; println(\"NPZ.jl installed\")"
        result = subprocess.run(['julia', '-e', check_script],
                              capture_output=True,
                              timeout=5)
        if result.returncode != 0:
            print("Installing NPZ.jl package...")
            install_cmd = 'using Pkg; Pkg.add("NPZ")'
            subprocess.run(['julia', '-e', install_cmd], timeout=60)

    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Julia not found or timed out!")
        print("Please install Julia from https://julialang.org/")
        print("Or set use_julia=False to use Python version")
        exit(1)

    # Run constrained decomposition
    results, biomarker_names = constrained_moments_decomposition(
        filepath="data.xlsx",
        rank=5,
        confidence_level=2.0,
        use_julia=True,
        verbose=True
    )

    print("\nDone! You can now:")
    print("  1. Compare constrained vs unconstrained results")
    print("  2. Analyze factor loadings with biological meaning")
    print("  3. Interpret constraints in terms of biomarker covariance")
