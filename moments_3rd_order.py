"""3rd-Order Moment Tensor Decomposition"""

import numpy as np
import pandas as pd
from itertools import permutations
import tensorly as tl
from tensorly.decomposition import parafac


def group_data_by_column(df, group_col='Group'):
    """Group data by column and extract biomarker data."""
    # Find biomarker columns (skip first 5 metadata columns)
    biomarker_cols = df.columns[5:].tolist()
    groups = df[group_col].unique()
    
    data_by_groups = []
    for group in groups:
        idx = df[group_col] == group
        group_data = df.loc[idx, biomarker_cols].fillna(df[biomarker_cols].mean()).astype(float)
        data_by_groups.append(group_data.values)
    
    return groups, data_by_groups


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
                for idx in set(permutations([i, j, k])):
                    M[idx] = moment
    
    return M


def decompose_tensor(M, rank=5):
    """Decompose symmetric 3rd-order tensor using CP decomposition."""
    tensor_norm = np.linalg.norm(M)
    if tensor_norm < 1e-10:
        raise ValueError("Tensor norm too small")
    
    T = tl.tensor(M / tensor_norm)
    factors = parafac(T, rank=rank, init='svd', random_state=42, n_iter_max=2000, tol=1e-8, verbose=0)
    
    # Average factors for symmetric tensor
    v_symmetric = np.mean(factors.factors, axis=0)
    
    # Normalize
    for r in range(rank):
        v_norm = np.linalg.norm(v_symmetric[:, r])
        if v_norm > 1e-10:
            v_symmetric[:, r] /= v_norm
    
    # Reconstruct
    recon_norm = tl.cp_to_tensor((factors.weights, [v_symmetric, v_symmetric, v_symmetric]))
    reconstruction = recon_norm * tensor_norm
    
    return {
        'factors': v_symmetric,
        'weights': factors.weights,
        'relative_error': float(np.linalg.norm(M - reconstruction) / tensor_norm)
    }


def main(filepath="data.xlsx", sheet_name="ATN_sharp", rank=5):
    """Main function to compute 3rd-order moments and decompose."""
    # Read data
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except FileNotFoundError:
        from cumulant_tensor_model import parse_csf_data
        X, biomarkers, metadata = parse_csf_data()
        df = pd.concat([metadata, pd.DataFrame(X, columns=biomarkers)], axis=1)
    
    groups, data_by_groups = group_data_by_column(df)
    decomps = []
    
    for idx, (group, X) in enumerate(zip(groups, data_by_groups)):
        M = compute_3rd_order_moments(X)
        try:
            decomp = decompose_tensor(M, rank=rank)
            decomp['group'] = group
            decomp['moment_tensor'] = M
            decomps.append(decomp)
            print(f"Group {idx+1}/{len(groups)}: {group} - Error: {decomp['relative_error']:.4f}")
        except Exception:
            print(f"Group {idx+1}/{len(groups)}: {group} - Failed")
            decomps.append({'group': group, 'relative_error': None})
    
    errors = [d['relative_error'] for d in decomps if d.get('relative_error') is not None]
    if errors:
        print(f"\nMean error: {np.mean(errors):.4f} | Min: {np.min(errors):.4f} | Max: {np.max(errors):.4f}")
    
    return decomps


if __name__ == "__main__":
    decomps = main()
