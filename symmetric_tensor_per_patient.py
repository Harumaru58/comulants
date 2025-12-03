"""
Symmetric Tensor Decomposition for Each Patient

This script creates a 28×28×28 symmetric tensor for each patient representing
third-order biomarker interactions, and performs symmetric tensor decomposition
using tensorly.
"""

import numpy as np
import pandas as pd
from cumulant_tensor_model import parse_csf_data
import warnings
warnings.filterwarnings('ignore')

# Import tensorly at module level
try:
    import tensorly as tl
    from tensorly.decomposition import parafac, tucker
    TENSORLY_AVAILABLE = True
except ImportError:
    TENSORLY_AVAILABLE = False
    tl = None


def create_symmetric_3d_tensor_per_patient(X, mode='interaction', normalize=True):
    """
    Create a 28×28×28 symmetric tensor for each patient.
    
    Parameters:
    -----------
    X : array (n_patients, n_biomarkers)
        Biomarker measurements for all patients
    mode : str
        Method to create the tensor:
        - 'interaction': Biomarker interaction tensor with population context
        - 'outer_product': Outer product x_i ⊗ x_i ⊗ x_i (fully symmetric, rank-1)
        - 'covariance_3d': Third-order covariance-like tensor
    normalize : bool
        Whether to normalize the tensor
        
    Returns:
    --------
    patient_tensors : array (n_patients, 28, 28, 28)
        Symmetric 3D tensor for each patient
    """
    n_patients, n_biomarkers = X.shape
    
    if n_biomarkers != 28:
        raise ValueError(f"Expected 28 biomarkers, got {n_biomarkers}")
    
    # Standardize data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std
    
    # Compute population-level statistics for context
    X_pop_mean = np.mean(X_normalized, axis=0)
    X_pop_cov = np.cov(X_normalized.T)
    
    patient_tensors = np.zeros((n_patients, n_biomarkers, n_biomarkers, n_biomarkers))
    
    for i in range(n_patients):
        x_i = X_normalized[i, :]  # (28,)
        deviation = x_i - X_pop_mean  # Deviation from population mean
        
        if mode == 'interaction':
            # Create tensor that captures third-order interactions
            # Use a combination of outer products with different biomarkers to create rank > 1
            # T[j,k,l] = Σ_m w_m * (x_i[j] - μ_j) * (x_i[k] - μ_k) * (x_i[l] - μ_l) * basis_m[j,k,l]
            # where basis_m are constructed from population structure
            
            # Create basis from population covariance eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(X_pop_cov)
            # Use top 5 eigenvectors as basis
            n_basis = min(5, n_biomarkers)
            basis_vectors = eigenvecs[:, -n_basis:]  # (28, n_basis)
            
            for j in range(n_biomarkers):
                for k in range(n_biomarkers):
                    for l in range(n_biomarkers):
                        # Sum over basis vectors to create higher-rank tensor
                        tensor_val = 0.0
                        for m in range(n_basis):
                            # Create basis tensor element
                            basis_elem = (basis_vectors[j, m] * 
                                         basis_vectors[k, m] * 
                                         basis_vectors[l, m])
                            # Weight by patient's projection onto this basis
                            patient_proj = np.dot(x_i, basis_vectors[:, m])
                            tensor_val += patient_proj * basis_elem
                        
                        # Add main interaction term
                        main_term = deviation[j] * deviation[k] * deviation[l] * 0.1
                        patient_tensors[i, j, k, l] = tensor_val + main_term
            
            # Make fully symmetric
            patient_tensors[i] = (patient_tensors[i] + 
                                 np.transpose(patient_tensors[i], (1, 2, 0)) +
                                 np.transpose(patient_tensors[i], (2, 0, 1)) +
                                 np.transpose(patient_tensors[i], (0, 2, 1)) +
                                 np.transpose(patient_tensors[i], (1, 0, 2)) +
                                 np.transpose(patient_tensors[i], (2, 1, 0))) / 6
        
        elif mode == 'outer_product':
            # Create fully symmetric tensor: x_i ⊗ x_i ⊗ x_i
            # This is rank-1, so decomposition with rank>1 will have high error
            for j in range(n_biomarkers):
                for k in range(n_biomarkers):
                    for l in range(n_biomarkers):
                        patient_tensors[i, j, k, l] = x_i[j] * x_i[k] * x_i[l]
        
        elif mode == 'covariance_3d':
            # Create tensor based on covariance structure
            # T[j,k,l] = Cov[j,k] * x_i[l] + Cov[j,l] * x_i[k] + Cov[k,l] * x_i[j]
            for j in range(n_biomarkers):
                for k in range(n_biomarkers):
                    for l in range(n_biomarkers):
                        patient_tensors[i, j, k, l] = (
                            X_pop_cov[j, k] * x_i[l] +
                            X_pop_cov[j, l] * x_i[k] +
                            X_pop_cov[k, l] * x_i[j]
                        ) / 3
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Normalize tensor to prevent numerical issues
        if normalize:
            tensor_norm = np.linalg.norm(patient_tensors[i])
            if tensor_norm > 1e-10:
                patient_tensors[i] = patient_tensors[i] / tensor_norm
    
    return patient_tensors


def decompose_symmetric_3d_tensor(tensor_3d, rank=5, method='cp', tol=1e-6):
    """
    Decompose a symmetric 28×28×28 tensor using tensorly.
    
    For a symmetric 3D tensor T:
    - CP: T ≈ Σᵣ λᵣ vᵣ ⊗ vᵣ ⊗ vᵣ (fully symmetric)
    - Tucker: T ≈ G ×₁ U ×₂ U ×₃ U (symmetric Tucker)
    
    Parameters:
    -----------
    tensor_3d : array (28, 28, 28)
        Symmetric 3D tensor
    rank : int
        Rank of decomposition
    method : str
        'cp' or 'tucker'
    tol : float
        Tolerance for convergence
        
    Returns:
    --------
    result : dict
        Decomposition factors and metrics
    """
    if not TENSORLY_AVAILABLE:
        raise ImportError("tensorly is required. Install with: pip install tensorly")
    
    T = tl.tensor(tensor_3d)
    tensor_norm = tl.norm(T)
    
    # Check if tensor is too small (numerical issues)
    if tensor_norm < 1e-10:
        raise ValueError("Tensor norm too small, likely numerical issues")
    
    if method == 'cp':
        # CP decomposition: T ≈ Σᵣ λᵣ vᵣ ⊗ vᵣ ⊗ vᵣ
        # Use multiple initializations and pick the best
        best_error = np.inf
        best_factors = None
        
        for init_method in ['svd', 'random']:
            try:
                factors = parafac(T, rank=rank, init=init_method, random_state=42, 
                                n_iter_max=1000, tol=tol, verbose=0)
                # Check reconstruction quality
                recon = tl.cp_to_tensor(factors)
                error = float(tl.norm(T - recon) / tl.norm(T))
                if error < best_error:
                    best_error = error
                    best_factors = factors
            except:
                continue
        
        if best_factors is None:
            # Last resort: random with more iterations
            factors = parafac(T, rank=rank, init='random', random_state=42, 
                            n_iter_max=2000, tol=tol*0.1, verbose=0)
        else:
            factors = best_factors
        
        # Extract factors
        v1 = factors.factors[0]  # (28, rank)
        v2 = factors.factors[1]  # (28, rank)
        v3 = factors.factors[2]  # (28, rank)
        
        # For symmetric tensor, average the factors
        v_symmetric = (v1 + v2 + v3) / 3
        
        # Normalize factors
        for r in range(rank):
            v_norm = np.linalg.norm(v_symmetric[:, r])
            if v_norm > 1e-10:
                v_symmetric[:, r] = v_symmetric[:, r] / v_norm
        
        # Reconstruct using symmetric factors
        reconstruction = tl.cp_to_tensor((factors.weights, 
                                         [v_symmetric, v_symmetric, v_symmetric]))
        
        result = {
            'method': 'cp',
            'weights': factors.weights,
            'factors': v_symmetric,  # (28, rank) - symmetric biomarker factors
            'reconstruction': reconstruction,
            'original_factors': [v1, v2, v3]
        }
        
    elif method == 'tucker':
        # Tucker decomposition with symmetric structure
        # T ≈ G ×₁ U ×₂ U ×₃ U where U is the same for all modes
        try:
            core, factors = tucker(T, rank=[rank, rank, rank], 
                                  init='svd', random_state=42, 
                                  n_iter_max=500, tol=tol, verbose=0)
        except:
            core, factors = tucker(T, rank=[rank, rank, rank], 
                                  init='random', random_state=42, 
                                  n_iter_max=500, tol=tol, verbose=0)
        
        # For symmetric tensor, average the factors
        u_symmetric = (factors[0] + factors[1] + factors[2]) / 3
        
        # Normalize factors
        for r in range(rank):
            u_norm = np.linalg.norm(u_symmetric[:, r])
            if u_norm > 1e-10:
                u_symmetric[:, r] = u_symmetric[:, r] / u_norm
        
        # Reconstruct
        reconstruction = tl.tucker_to_tensor((core, 
                                              [u_symmetric, u_symmetric, u_symmetric]))
        
        result = {
            'method': 'tucker',
            'core': core,  # (rank, rank, rank)
            'factors': u_symmetric,  # (28, rank)
            'reconstruction': reconstruction,
            'original_factors': factors
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute reconstruction error (relative)
    reconstruction_error = tl.norm(T - reconstruction)
    relative_error = reconstruction_error / tensor_norm
    
    result['reconstruction_error'] = float(reconstruction_error)
    result['relative_error'] = float(relative_error)
    result['tensor_norm'] = float(tensor_norm)
    
    return result


def analyze_patient_tensors(patient_tensors, patient_indices=None):
    """
    Analyze and decompose symmetric tensors for multiple patients.
    
    Parameters:
    -----------
    patient_tensors : array (n_patients, 28, 28, 28)
        Symmetric tensors for each patient
    patient_indices : array-like, optional
        Which patients to analyze. If None, analyze all.
        
    Returns:
    --------
    results : dict
        Decomposition results for each patient
    """
    n_patients = patient_tensors.shape[0]
    
    if patient_indices is None:
        patient_indices = range(n_patients)
    
    results = {
        'patient_decompositions': {},
        'summary': {
            'n_patients': len(patient_indices),
            'tensor_shape': (28, 28, 28),
            'reconstruction_errors': [],
            'factors_shapes': []
        }
    }
    
    print(f"Decomposing symmetric tensors for {len(patient_indices)} patients...")
    
    for idx, patient_idx in enumerate(patient_indices):
        if (idx + 1) % 10 == 0:
            print(f"  Processing patient {idx + 1}/{len(patient_indices)}...")
        
        tensor_3d = patient_tensors[patient_idx]
        
        # Try both CP and Tucker, use the one with better reconstruction
        try:
            # Try Tucker first (often better for symmetric tensors)
            tucker_result = decompose_symmetric_3d_tensor(tensor_3d, rank=5, method='tucker', tol=1e-5)
            # Also try CP
            cp_result = decompose_symmetric_3d_tensor(tensor_3d, rank=5, method='cp', tol=1e-5)
            
            # Use the method with lower error
            if tucker_result['relative_error'] < cp_result['relative_error']:
                best_result = tucker_result
                best_method = 'tucker'
            else:
                best_result = cp_result
                best_method = 'cp'
            
            results['patient_decompositions'][patient_idx] = {
                'best': best_result,
                'method': best_method,
                'cp': cp_result,
                'tucker': tucker_result
            }
            # Use relative error from best method
            results['summary']['reconstruction_errors'].append(best_result['relative_error'])
            results['summary']['factors_shapes'].append(best_result['factors'].shape)
        except Exception as e:
            print(f"  Warning: Failed to decompose patient {patient_idx}: {e}")
            results['patient_decompositions'][patient_idx] = {'error': str(e)}
    
    # Compute summary statistics
    if results['summary']['reconstruction_errors']:
        results['summary']['mean_error'] = np.mean(results['summary']['reconstruction_errors'])
        results['summary']['std_error'] = np.std(results['summary']['reconstruction_errors'])
        results['summary']['min_error'] = np.min(results['summary']['reconstruction_errors'])
        results['summary']['max_error'] = np.max(results['summary']['reconstruction_errors'])
    
    return results


def main():
    """Main function to run symmetric tensor decomposition per patient."""
    
    print("="*70)
    print("Symmetric 28×28×28 Tensor Decomposition Per Patient")
    print("="*70)
    
    # Load data
    print("\n1. Loading data from CSV...")
    X, biomarkers, metadata = parse_csf_data()
    print(f"   Data shape: {X.shape} (patients × biomarkers)")
    print(f"   Number of patients: {X.shape[0]}")
    print(f"   Number of biomarkers: {X.shape[1]}")
    
    if X.shape[1] != 28:
        print(f"   Warning: Expected 28 biomarkers, got {X.shape[1]}")
        print(f"   Using first 28 biomarkers...")
        X = X[:, :28]
        biomarkers = biomarkers[:28]
    
    # Create symmetric 3D tensors for each patient
    print("\n2. Creating symmetric 28×28×28 tensors for each patient...")
    print("   Mode: interaction (with population context for better decomposition)")
    
    patient_tensors = create_symmetric_3d_tensor_per_patient(X, mode='interaction', normalize=True)
    print(f"   Patient tensors shape: {patient_tensors.shape}")
    print(f"   Memory usage: {patient_tensors.nbytes / 1024**2:.2f} MB")
    
    # Check tensor norms
    tensor_norms = [np.linalg.norm(patient_tensors[i]) for i in range(min(10, X.shape[0]))]
    print(f"   Tensor norms (first 10 patients): min={np.min(tensor_norms):.4f}, "
          f"max={np.max(tensor_norms):.4f}, mean={np.mean(tensor_norms):.4f}")
    
    # Analyze a subset of patients first (for testing)
    print("\n3. Decomposing symmetric tensors...")
    print("   Analyzing first 10 patients as demonstration...")
    
    n_demo = min(10, X.shape[0])
    demo_results = analyze_patient_tensors(patient_tensors, patient_indices=range(n_demo))
    
    # Print summary
    print("\n" + "="*70)
    print("Summary Statistics (first 10 patients):")
    print("="*70)
    summary = demo_results['summary']
    print(f"Number of patients analyzed: {summary['n_patients']}")
    print(f"Tensor shape per patient: {summary['tensor_shape']}")
    if summary['reconstruction_errors']:
        print(f"\nRelative Reconstruction Errors (CP decomposition):")
        print(f"  Mean: {summary['mean_error']:.6f} ({summary['mean_error']*100:.4f}%)")
        print(f"  Std:  {summary['std_error']:.6f}")
        print(f"  Min:  {summary['min_error']:.6f} ({summary['min_error']*100:.4f}%)")
        print(f"  Max:  {summary['max_error']:.6f} ({summary['max_error']*100:.4f}%)")
    
    # Example: Show factors for first patient
    if 0 in demo_results['patient_decompositions']:
        patient_0_decomp = demo_results['patient_decompositions'][0]
        if 'best' in patient_0_decomp:
            patient_0_result = patient_0_decomp['best']
            method_used = patient_0_decomp.get('method', 'unknown')
            print(f"\nExample: Patient 0 decomposition (best method: {method_used}):")
            print(f"  Biomarker factors shape: {patient_0_result['factors'].shape}")
            print(f"  Tensor norm: {patient_0_result['tensor_norm']:.6f}")
            print(f"  Relative reconstruction error: {patient_0_result['relative_error']:.6f} "
                  f"({patient_0_result['relative_error']*100:.4f}%)")
            if 'cp' in patient_0_decomp and 'tucker' in patient_0_decomp:
                print(f"  CP error: {patient_0_decomp['cp']['relative_error']:.6f}, "
                      f"Tucker error: {patient_0_decomp['tucker']['relative_error']:.6f}")
            print(f"  Top 5 biomarker loadings (first factor):")
            top_5_idx = np.argsort(np.abs(patient_0_result['factors'][:, 0]))[-5:][::-1]
            for idx in top_5_idx:
                print(f"    {biomarkers[idx]}: {patient_0_result['factors'][idx, 0]:.4f}")
        elif 'error' in patient_0_decomp:
            print(f"\nExample: Patient 0 decomposition failed: {patient_0_decomp['error']}")
    
    # Option to analyze all patients
    print("\n" + "="*70)
    print("To analyze all patients, uncomment the following in the code:")
    print("="*70)
    print("# all_results = analyze_patient_tensors(patient_tensors)")
    print("# Save results if needed")
    
    return patient_tensors, demo_results


if __name__ == "__main__":
    patient_tensors, results = main()

