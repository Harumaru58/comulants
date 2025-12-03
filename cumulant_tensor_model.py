import numpy as np
from scipy import stats
from scipy.special import factorial
import pandas as pd
import os

# Load and parse the CSF biomarker data from CSV
def parse_csf_data(csv_path=None):
    """
    Parse the CSF proteomics data from the CSV file.
    
    Parameters:
    -----------
    csv_path : str, optional
        Path to CSV file. If None, uses default filename.
    
    Returns:
    --------
    data : array (n_patients, n_biomarkers)
        Biomarker measurements
    biomarkers : list
        List of biomarker names
    metadata : DataFrame
        Patient metadata (Group, ApoE Pheno, sex, age, Sample)
    """
    if csv_path is None:
        csv_path = "NIpanel_msclin_ATNsharp_20241205(1).csv"
    
    # Read the header - it's split across two lines
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    
    # Combine the two header lines
    # First line is: Group;"ApoE
    # Second line starts with: Pheno";sex;...
    # Combine them: Group;ApoE Pheno;sex;...
    if 'ApoE' in first_line:
        # Header is split - combine first and second line
        first_clean = first_line.replace('"ApoE', '').rstrip()
        second_clean = second_line.replace('Pheno";', 'ApoE Pheno;').lstrip()
        header_line = first_clean + second_clean
    else:
        # Normal single-line header
        header_line = first_line
    
    # Parse header manually
    header_parts = [h.strip().strip('"') for h in header_line.split(';')]
    # Remove empty strings
    header_parts = [h for h in header_parts if h]
    
    # Read CSV with semicolon delimiter, skipping the first line (header)
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8', skiprows=1, header=None)
    
    # Set column names from our manually parsed header
    n_cols = len(df.columns)
    n_header = len(header_parts)
    
    if n_cols == n_header:
        df.columns = header_parts
    elif n_cols > n_header:
        # More columns than header - pad header
        df.columns = header_parts + [f'col_{i}' for i in range(n_cols - n_header)]
    else:
        # Fewer columns than header - truncate header
        df.columns = header_parts[:n_cols]
    
    # Biomarker columns (excluding metadata columns)
    # Metadata columns are typically the first 5
    metadata_col_names = ['Group', 'ApoE Pheno', 'sex', 'age at LP', 'Sample']
    # Try to match metadata columns (case-insensitive, handle variations)
    metadata_cols = []
    for meta_name in metadata_col_names:
        for col in df.columns:
            if meta_name.lower() in col.lower() or col.lower() in meta_name.lower():
                if col not in metadata_cols:
                    metadata_cols.append(col)
                    break
    
    # If we couldn't find metadata columns, assume first 5 are metadata
    if len(metadata_cols) < 3:
        metadata_cols = list(df.columns[:5])
    
    # Get all columns that are not metadata
    biomarker_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Extract metadata
    try:
        metadata = df[metadata_cols].copy()
    except:
        metadata = df.iloc[:, :len(metadata_cols)].copy()
        metadata.columns = metadata_cols[:len(metadata.columns)]
    
    # Extract biomarker data
    biomarker_data = df[biomarker_cols].copy()
    
    # Convert comma decimal separators to dots and convert to float
    for col in biomarker_cols:
        if biomarker_data[col].dtype == 'object':
            # Replace comma with dot and convert to float
            # Handle any string values
            biomarker_data[col] = biomarker_data[col].astype(str).str.replace(',', '.').replace('nan', 'NaN')
            # Convert to numeric, coercing errors to NaN
            biomarker_data[col] = pd.to_numeric(biomarker_data[col], errors='coerce')
    
    # Drop any rows with all NaN biomarker values
    valid_rows = ~biomarker_data.isna().all(axis=1)
    biomarker_data = biomarker_data[valid_rows].copy()
    metadata = metadata[valid_rows].copy()
    
    # Fill remaining NaN with column mean
    biomarker_data = biomarker_data.fillna(biomarker_data.mean())
    
    # Convert to numpy array
    data = biomarker_data.values.astype(float)
    
    # Clean biomarker names (remove brackets, spaces)
    biomarkers = [col.replace('[', '_').replace(']', '').replace(' ', '_') for col in biomarker_cols]
    
    return data, biomarkers, metadata


class LatentCumulantTensorModel:
    """
    Generative model for patient × biomarker × cumulant tensor.
    
    Model structure:
    - Each patient i has a latent biomarker state vector z_i ~ p(z | θ_i)
    - θ_i = (κ₁, κ₂, κ₃, κ₄) are cumulants parameterizing patient's distribution
    - Observed biomarker values x_ij are noisy observations of latent states
    
    The tensor T[i,j,k] represents the k-th cumulant of biomarker j for patient i.
    """
    
    def __init__(self, n_cumulants=4):
        self.n_cumulants = n_cumulants
        self.cumulant_names = ['κ₁ (mean)', 'κ₂ (variance)', 'κ₃ (skewness)', 'κ₄ (kurtosis)']
    
    def fit(self, X, n_bootstrap=100):
        """
        Estimate cumulants from observed data using method of moments.
        
        Parameters:
        -----------
        X : array (n_patients, n_biomarkers)
            Observed biomarker measurements
        n_bootstrap : int
            Number of bootstrap samples for uncertainty estimation
            
        Returns:
        --------
        tensor : array (n_patients, n_biomarkers, n_cumulants)
        """
        n_patients, n_biomarkers = X.shape
        
        # Standardize within biomarkers for numerical stability
        self.biomarker_means = np.mean(X, axis=0)
        self.biomarker_stds = np.std(X, axis=0) + 1e-8
        X_std = (X - self.biomarker_means) / self.biomarker_stds
        
        # Initialize tensor
        tensor = np.zeros((n_patients, n_biomarkers, self.n_cumulants))
        
        # Estimate cumulants for each patient-biomarker pair
        # Using a sliding window / local estimation approach
        for i in range(n_patients):
            for j in range(n_biomarkers):
                # For single observation, use population-informed priors
                # Estimate via empirical Bayes: shrink toward global estimates
                
                x_ij = X_std[i, j]
                
                # Global statistics for biomarker j (across patients)
                x_j = X_std[:, j]
                
                # κ₁: Mean (shrunk toward patient's deviation from biomarker mean)
                tensor[i, j, 0] = x_ij
                
                # κ₂: Variance (use squared deviation, shrunk toward population)
                pop_var = np.var(x_j)
                tensor[i, j, 1] = 0.5 * (x_ij ** 2) + 0.5 * pop_var
                
                # κ₃: Skewness contribution (third moment)
                pop_skew = stats.skew(x_j)
                tensor[i, j, 2] = 0.5 * (x_ij ** 3) + 0.5 * pop_skew
                
                # κ₄: Kurtosis contribution (fourth moment - 3*var²)
                pop_kurt = stats.kurtosis(x_j, fisher=True)
                tensor[i, j, 3] = 0.5 * (x_ij ** 4 - 3) + 0.5 * pop_kurt
        
        self.tensor = tensor
        return tensor
    
    def fit_generative(self, X, latent_dim=3, n_iter=100):
        """
        Fit a proper generative latent variable model.
        
        Generative Model:
        - Each patient i has a latent biomarker state z_i ~ p(z | κ_i)
        - The distribution p(z | κ_i) is parameterized by cumulants κ_i = (κ₁, κ₂, κ₃, κ₄)
        - Observed biomarkers x_ij are generated as: x_ij = f(z_i, β_j) + ε_ij
        - The tensor T[i,j,k] stores the k-th cumulant κ_k for biomarker j of patient i
        
        This creates a 3-way symmetric tensor: patients × biomarkers × cumulants
        where cumulants parameterize the distribution of each patient's biomarker state.
        
        Parameters:
        -----------
        X : array (n_patients, n_biomarkers)
            Observed biomarker measurements
        latent_dim : int
            Dimensionality of latent biomarker state
        n_iter : int
            Number of iterations (for future iterative refinement)
            
        Returns:
        --------
        tensor : array (n_patients, n_biomarkers, n_cumulants)
            3-way tensor where T[i,j,k] is the k-th cumulant of biomarker j for patient i
        """
        n_patients, n_biomarkers = X.shape
        
        # Standardize
        self.biomarker_means = np.mean(X, axis=0)
        self.biomarker_stds = np.std(X, axis=0) + 1e-8
        X_std = (X - self.biomarker_means) / self.biomarker_stds
        
        # Initialize latent factors via SVD
        # This gives us an initial estimate of patient latent states
        U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
        Z = U[:, :latent_dim] * S[:latent_dim]  # Patient latent states z_i
        W = Vt[:latent_dim, :].T  # Biomarker loadings β_j
        
        # Initialize the 3-way tensor: patients × biomarkers × cumulants
        tensor = np.zeros((n_patients, n_biomarkers, self.n_cumulants))
        
        # For each patient, estimate cumulants that parameterize their biomarker state distribution
        for i in range(n_patients):
            z_i = Z[i, :]  # Latent biomarker state for patient i
            
            # Estimate global cumulants of the latent state distribution
            # These characterize p(z_i | κ_i) - the distribution of patient i's biomarker state
            z_mean = np.mean(z_i)
            z_var = np.var(z_i) + 1e-8
            z_std = np.sqrt(z_var)
            
            # Higher moments of latent state
            z_centered = z_i - z_mean
            z_skew = np.mean(z_centered ** 3) / (z_std ** 3 + 1e-8)
            z_kurt = np.mean(z_centered ** 4) / (z_var ** 2 + 1e-8) - 3
            
            # For each biomarker, compute cumulants that parameterize the distribution
            # of that biomarker given the patient's latent state
            for j in range(n_biomarkers):
                w_j = W[j, :]  # Loading for biomarker j
                
                # Project latent state onto biomarker direction
                # This gives the expected value of biomarker j given patient i's state
                proj = np.dot(z_i, w_j)
                
                # Residual captures patient-specific deviation from the latent structure
                resid = X_std[i, j] - proj
                
                # Compute cumulants that parameterize the biomarker distribution
                # These cumulants characterize the distribution of biomarker j for patient i
                
                # κ₁: First cumulant (mean) - expected value given latent state
                tensor[i, j, 0] = proj
                
                # κ₂: Second cumulant (variance) - uncertainty in biomarker given latent state
                tensor[i, j, 1] = max(resid ** 2 + 0.1, 1e-6)  # Ensure positive
                
                # κ₃: Third cumulant (skewness) - asymmetry in distribution
                # Normalized by variance^(3/2) to get standardized skewness
                tensor[i, j, 2] = resid ** 3 / (tensor[i, j, 1] ** 1.5 + 1e-8)
                
                # κ₄: Fourth cumulant (excess kurtosis) - tail heaviness
                # Normalized by variance^2, minus 3 for excess kurtosis
                tensor[i, j, 3] = (resid ** 4) / (tensor[i, j, 1] ** 2 + 1e-8) - 3
        
        self.tensor = tensor
        self.latent_states = Z
        self.loadings = W
        return tensor
    
    def fit_mixture(self, X, n_components=3):
        """
        Fit a Gaussian mixture model and derive cumulants from mixture parameters.
        
        Each patient's biomarker distribution is modeled as a mixture,
        with cumulants derived from the mixture moments.
        """
        from sklearn.mixture import GaussianMixture
        
        n_patients, n_biomarkers = X.shape
        X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        
        tensor = np.zeros((n_patients, n_biomarkers, self.n_cumulants))
        
        # Fit GMM to each biomarker
        for j in range(n_biomarkers):
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(X_std[:, j].reshape(-1, 1))
            
            # Get responsibilities (soft assignments)
            resp = gmm.predict_proba(X_std[:, j].reshape(-1, 1))
            
            for i in range(n_patients):
                x_ij = X_std[i, j]
                r_i = resp[i]  # Component responsibilities
                
                # Mixture cumulants (weighted by responsibilities)
                means = gmm.means_.flatten()
                vars = gmm.covariances_.flatten()
                
                # κ₁: mixture mean
                tensor[i, j, 0] = np.sum(r_i * means)
                
                # κ₂: mixture variance (includes between-component variance)
                tensor[i, j, 1] = np.sum(r_i * (vars + means**2)) - tensor[i, j, 0]**2
                
                # κ₃, κ₄: higher cumulants from deviation
                dev = x_ij - tensor[i, j, 0]
                tensor[i, j, 2] = dev ** 3 / (tensor[i, j, 1] ** 1.5 + 1e-8)
                tensor[i, j, 3] = dev ** 4 / (tensor[i, j, 1] ** 2 + 1e-8) - 3
        
        self.tensor = tensor
        return tensor
    
    def generate_samples(self, n_samples=100):
        """
        Generate synthetic data from the fitted cumulant tensor.
        
        Uses Edgeworth expansion to sample from distribution with given cumulants.
        """
        n_patients, n_biomarkers, _ = self.tensor.shape
        samples = np.zeros((n_patients, n_biomarkers, n_samples))
        
        for i in range(n_patients):
            for j in range(n_biomarkers):
                κ = self.tensor[i, j, :]
                
                # Generate base Gaussian samples
                z = np.random.randn(n_samples)
                
                # Edgeworth expansion correction
                # x ≈ μ + σ(z + γ₁/6(z²-1) + γ₂/24(z³-3z))
                mu, var, skew, kurt = κ
                sigma = np.sqrt(max(var, 1e-8))
                
                x = mu + sigma * (z + skew/6 * (z**2 - 1) + kurt/24 * (z**3 - 3*z))
                samples[i, j, :] = x
        
        return samples
    
    def decompose_cp(self, rank=5):
        """
        CP decomposition of the cumulant tensor.
        
        T ≈ Σᵣ aᵣ ⊗ bᵣ ⊗ cᵣ
        
        where:
        - aᵣ: patient factors (patient phenotypes)
        - bᵣ: biomarker factors (biomarker modules)  
        - cᵣ: cumulant factors (distributional signatures)
        """
        try:
            import tensorly as tl
            from tensorly.decomposition import parafac
        except ImportError:
            raise ImportError("tensorly is required for CP decomposition. Install with: pip install tensorly")
        
        # Fit CP decomposition
        factors = parafac(self.tensor, rank=rank, init='random', random_state=42)
        
        weights = factors.weights
        patient_factors = factors.factors[0]  # (n_patients, rank)
        biomarker_factors = factors.factors[1]  # (n_biomarkers, rank)
        cumulant_factors = factors.factors[2]  # (n_cumulants, rank)
        
        return {
            'weights': weights,
            'patient_factors': patient_factors,
            'biomarker_factors': biomarker_factors,
            'cumulant_factors': cumulant_factors
        }
    
    def create_symmetric_biomarker_tensor(self, X=None, mode='covariance'):
        """
        Create a symmetric 28×28 tensor from biomarker interactions.
        
        Parameters:
        -----------
        X : array (n_patients, n_biomarkers), optional
            Biomarker data. If None, uses self.tensor if available.
        mode : str
            How to compute interactions:
            - 'covariance': Covariance matrix of biomarkers
            - 'correlation': Correlation matrix of biomarkers
            - 'cumulant_mean': Mean cumulant interactions across patients
            - 'cumulant_all': All cumulants (28×28×4 tensor)
            
        Returns:
        --------
        symmetric_tensor : array
            Symmetric tensor of shape (28, 28) or (28, 28, n_cumulants)
        """
        if X is None:
            if not hasattr(self, 'tensor'):
                raise ValueError("No data provided and no tensor available. Call fit() or fit_generative() first.")
            n_biomarkers = self.tensor.shape[1]
        else:
            # Standardize if needed
            if not hasattr(self, 'biomarker_means'):
                self.biomarker_means = np.mean(X, axis=0)
                self.biomarker_stds = np.std(X, axis=0) + 1e-8
            X_std = (X - self.biomarker_means) / self.biomarker_stds
            n_biomarkers = X.shape[1]
        
        if mode == 'covariance':
            if X is None:
                raise ValueError("X required for covariance mode")
            # Compute covariance matrix (symmetric 28×28)
            symmetric_tensor = np.cov(X_std.T)
            
        elif mode == 'correlation':
            if X is None:
                raise ValueError("X required for correlation mode")
            # Compute correlation matrix (symmetric 28×28)
            symmetric_tensor = np.corrcoef(X_std.T)
            
        elif mode == 'cumulant_mean':
            if not hasattr(self, 'tensor'):
                raise ValueError("No tensor available. Call fit() or fit_generative() first.")
            # Average cumulants across patients, then compute interactions
            # Shape: (n_biomarkers, n_cumulants)
            mean_cumulants = np.mean(self.tensor, axis=0)
            # Create symmetric tensor from outer products
            # For each cumulant, compute biomarker × biomarker interactions
            symmetric_tensor = np.zeros((n_biomarkers, n_biomarkers, self.n_cumulants))
            for k in range(self.n_cumulants):
                # Outer product of cumulant k across biomarkers
                symmetric_tensor[:, :, k] = np.outer(mean_cumulants[:, k], mean_cumulants[:, k])
            
        elif mode == 'cumulant_all':
            if not hasattr(self, 'tensor'):
                raise ValueError("No tensor available. Call fit() or fit_generative() first.")
            # For each cumulant, compute symmetric biomarker interaction matrix
            symmetric_tensor = np.zeros((n_biomarkers, n_biomarkers, self.n_cumulants))
            for k in range(self.n_cumulants):
                # Compute covariance-like matrix for cumulant k across patients
                cumulant_data = self.tensor[:, :, k]  # (n_patients, n_biomarkers)
                symmetric_tensor[:, :, k] = np.cov(cumulant_data.T)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return symmetric_tensor
    
    def decompose_symmetric(self, symmetric_tensor, rank=5, method='cp'):
        """
        Perform symmetric tensor decomposition using tensorly.
        
        For a symmetric tensor T of shape (n, n) or (n, n, k):
        - CP decomposition: T ≈ Σᵣ λᵣ vᵣ ⊗ vᵣ (for 2D) or T ≈ Σᵣ λᵣ vᵣ ⊗ vᵣ ⊗ cᵣ (for 3D)
        
        Parameters:
        -----------
        symmetric_tensor : array
            Symmetric tensor of shape (28, 28) or (28, 28, k)
        rank : int
            Rank of the decomposition
        method : str
            Decomposition method: 'cp' (CANDECOMP/PARAFAC) or 'tucker'
            
        Returns:
        --------
        result : dict
            Decomposition factors and weights
        """
        try:
            import tensorly as tl
            from tensorly.decomposition import parafac, tucker
        except ImportError:
            raise ImportError("tensorly is required for symmetric decomposition. Install with: pip install tensorly")
        
        # Convert to tensorly tensor
        T = tl.tensor(symmetric_tensor)
        
        if len(symmetric_tensor.shape) == 2:
            # 2D symmetric matrix (28×28)
            # For symmetric matrix, we can use eigenvalue decomposition or CP
            if method == 'cp':
                # CP decomposition: T ≈ Σᵣ λᵣ vᵣ ⊗ vᵣ
                # For symmetric matrix, this is essentially eigendecomposition
                factors = parafac(T, rank=rank, init='random', random_state=42)
                # Extract symmetric factors
                v = factors.factors[0]  # (28, rank)
                # For symmetric decomposition, both factors should be the same
                result = {
                    'weights': factors.weights,
                    'factors': v,  # (28, rank) - biomarker factors
                    'reconstruction': tl.cp_to_tensor(factors)
                }
            else:
                raise ValueError(f"Method {method} not supported for 2D tensors")
                
        elif len(symmetric_tensor.shape) == 3:
            # 3D symmetric tensor (28×28×k)
            if method == 'cp':
                # CP decomposition: T ≈ Σᵣ λᵣ vᵣ ⊗ vᵣ ⊗ cᵣ
                factors = parafac(T, rank=rank, init='random', random_state=42)
                v = factors.factors[0]  # (28, rank) - biomarker factors (symmetric)
                c = factors.factors[2]  # (k, rank) - third mode factors
                result = {
                    'weights': factors.weights,
                    'biomarker_factors': v,  # (28, rank)
                    'third_mode_factors': c,  # (k, rank)
                    'reconstruction': tl.cp_to_tensor(factors)
                }
            elif method == 'tucker':
                # Tucker decomposition
                core, factors = tucker(T, rank=[rank, rank, min(rank, T.shape[2])], 
                                       init='random', random_state=42)
                result = {
                    'core': core,
                    'biomarker_factors': factors[0],  # (28, rank)
                    'third_mode_factors': factors[2],  # (k, rank)
                    'reconstruction': tl.tucker_to_tensor((core, factors))
                }
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            raise ValueError(f"Unsupported tensor shape: {symmetric_tensor.shape}")
        
        # Compute reconstruction error
        result['reconstruction_error'] = tl.norm(T - result['reconstruction']) / tl.norm(T)
        
        return result


# Demo with the CSF data
if __name__ == "__main__":
    # Load data from CSV
    print("Loading data from CSV...")
    X, biomarkers, metadata = parse_csf_data()
    print(f"Data shape: {X.shape} (patients × biomarkers)")
    print(f"Number of patients: {X.shape[0]}")
    print(f"Number of biomarkers: {X.shape[1]}")
    print(f"\nBiomarker names (first 10): {biomarkers[:10]}")
    
    # Fit the model
    model = LatentCumulantTensorModel(n_cumulants=4)
    
    # Method 1: Direct cumulant estimation
    print("\n" + "="*60)
    print("Method 1: Direct cumulant estimation")
    print("="*60)
    tensor1 = model.fit(X)
    print(f"Tensor shape (direct): {tensor1.shape}")
    print(f"  - Patients: {tensor1.shape[0]}")
    print(f"  - Biomarkers: {tensor1.shape[1]}")
    print(f"  - Cumulants: {tensor1.shape[2]}")
    
    # Method 2: Generative latent variable model
    print("\n" + "="*60)
    print("Method 2: Generative latent variable model")
    print("="*60)
    tensor2 = model.fit_generative(X, latent_dim=3)
    print(f"Tensor shape (generative): {tensor2.shape}")
    print(f"  - Patients: {tensor2.shape[0]}")
    print(f"  - Biomarkers: {tensor2.shape[1]}")
    print(f"  - Cumulants: {tensor2.shape[2]}")
    
    # Generate synthetic samples
    print("\n" + "="*60)
    print("Generating synthetic samples...")
    print("="*60)
    samples = model.generate_samples(n_samples=50)
    print(f"Generated samples shape: {samples.shape}")
    
    # Example: inspect cumulants for patient 0, biomarker 0
    print("\n" + "="*60)
    print(f"Example: Patient 0, {biomarkers[0]} cumulants (generative model):")
    print("="*60)
    for k, name in enumerate(model.cumulant_names):
        print(f"  {name}: {tensor2[0, 0, k]:.4f}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Tensor Summary Statistics:")
    print("="*60)
    print(f"Mean cumulant values across all patients and biomarkers:")
    for k, name in enumerate(model.cumulant_names):
        print(f"  {name}: {np.mean(tensor2[:, :, k]):.4f} ± {np.std(tensor2[:, :, k]):.4f}")
    
    # Try CP decomposition if tensorly is available
    print("\n" + "="*60)
    print("CP Decomposition (if tensorly available):")
    print("="*60)
    try:
        cp_result = model.decompose_cp(rank=5)
        print(f"CP decomposition successful!")
        print(f"  Patient factors shape: {cp_result['patient_factors'].shape}")
        print(f"  Biomarker factors shape: {cp_result['biomarker_factors'].shape}")
        print(f"  Cumulant factors shape: {cp_result['cumulant_factors'].shape}")
    except ImportError:
        print("  tensorly not installed. Install with: pip install tensorly")
    
    # Symmetric tensor decomposition (28×28 biomarker interactions)
    print("\n" + "="*60)
    print("Symmetric Tensor Decomposition (28×28 biomarker interactions):")
    print("="*60)
    try:
        # Create symmetric biomarker tensor
        print("Creating symmetric biomarker tensor...")
        sym_tensor_cov = model.create_symmetric_biomarker_tensor(X, mode='covariance')
        print(f"  Covariance matrix shape: {sym_tensor_cov.shape}")
        
        sym_tensor_cum = model.create_symmetric_biomarker_tensor(X, mode='cumulant_all')
        print(f"  Cumulant tensor shape: {sym_tensor_cum.shape}")
        
        # Decompose 2D symmetric matrix (28×28)
        print("\nDecomposing 2D symmetric matrix (covariance):")
        sym_result_2d = model.decompose_symmetric(sym_tensor_cov, rank=5, method='cp')
        print(f"  Decomposition successful!")
        print(f"  Biomarker factors shape: {sym_result_2d['factors'].shape}")
        print(f"  Reconstruction error: {sym_result_2d['reconstruction_error']:.6f}")
        
        # Decompose 3D symmetric tensor (28×28×4)
        print("\nDecomposing 3D symmetric tensor (cumulant interactions):")
        sym_result_3d = model.decompose_symmetric(sym_tensor_cum, rank=5, method='cp')
        print(f"  Decomposition successful!")
        print(f"  Biomarker factors shape: {sym_result_3d['biomarker_factors'].shape}")
        print(f"  Cumulant factors shape: {sym_result_3d['third_mode_factors'].shape}")
        print(f"  Reconstruction error: {sym_result_3d['reconstruction_error']:.6f}")
        
        # Also try Tucker decomposition for 3D tensor
        print("\nTucker decomposition of 3D symmetric tensor:")
        tucker_result = model.decompose_symmetric(sym_tensor_cum, rank=5, method='tucker')
        print(f"  Tucker decomposition successful!")
        print(f"  Core tensor shape: {tucker_result['core'].shape}")
        print(f"  Biomarker factors shape: {tucker_result['biomarker_factors'].shape}")
        print(f"  Cumulant factors shape: {tucker_result['third_mode_factors'].shape}")
        print(f"  Reconstruction error: {tucker_result['reconstruction_error']:.6f}")
        
    except ImportError:
        print("  tensorly not installed. Install with: pip install tensorly")
    except Exception as e:
        print(f"  Error in symmetric decomposition: {e}")

