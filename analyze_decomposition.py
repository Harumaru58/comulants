"""
Analysis Tools for Group-Level Tensor Decomposition Results

This script provides functions to analyze and compare tensor decompositions
across patient groups to identify biomarker interaction patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import seaborn as sns


def compare_factors_across_groups(decomps, top_n=10):
    """
    Compare factor loadings across groups to identify:
    - Shared biomarker patterns
    - Group-specific patterns
    - Biomarker modules
    
    Parameters:
    -----------
    decomps : list of dict
        Results from moments_3rd_order.py or moments_4th_order.py
    top_n : int
        Number of top biomarkers to show per component
        
    Returns:
    --------
    analysis : dict
        Contains comparisons, similarities, and biomarker rankings
    """
    # Extract factors for each group
    group_factors = {}
    for decomp in decomps:
        if decomp.get('factors') is not None:
            group_factors[decomp['group']] = decomp['factors']
    
    # 1. Top biomarkers per component for each group
    top_biomarkers = {}
    for group, factors in group_factors.items():
        top_biomarkers[group] = {}
        for r in range(factors.shape[1]):
            # Get top N biomarkers for component r
            top_idx = np.argsort(np.abs(factors[:, r]))[-top_n:][::-1]
            top_biomarkers[group][f'component_{r}'] = {
                'indices': top_idx.tolist(),
                'loadings': factors[top_idx, r].tolist()
            }
    
    # 2. Compute similarity between groups (cosine similarity of factors)
    groups = list(group_factors.keys())
    similarity_matrix = np.zeros((len(groups), len(groups)))
    
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Compare first component (most important)
                sim = 1 - cosine(group_factors[group1][:, 0], group_factors[group2][:, 0])
                similarity_matrix[i, j] = sim
    
    # 3. Identify shared vs group-specific patterns
    # Shared: biomarkers that appear in top N across multiple groups
    all_top_biomarkers = set()
    for group_data in top_biomarkers.values():
        for comp_data in group_data.values():
            all_top_biomarkers.update(comp_data['indices'])
    
    biomarker_frequency = {idx: 0 for idx in all_top_biomarkers}
    for group_data in top_biomarkers.values():
        seen_in_group = set()
        for comp_data in group_data.values():
            seen_in_group.update(comp_data['indices'])
        for idx in seen_in_group:
            biomarker_frequency[idx] += 1
    
    # Shared biomarkers (appear in multiple groups)
    shared_biomarkers = [idx for idx, freq in biomarker_frequency.items() if freq >= 2]
    group_specific = {idx: freq for idx, freq in biomarker_frequency.items() if freq == 1}
    
    return {
        'top_biomarkers': top_biomarkers,
        'similarity_matrix': similarity_matrix,
        'group_names': groups,
        'shared_biomarkers': shared_biomarkers,
        'biomarker_frequency': biomarker_frequency
    }


def analyze_component_structure(decomps, biomarker_names=None):
    """
    Analyze the structure of components across groups.
    
    Parameters:
    -----------
    decomps : list of dict
        Decomposition results
    biomarker_names : list, optional
        Names of biomarkers for interpretation
        
    Returns:
    --------
    analysis : dict
        Component analysis results
    """
    results = {}
    
    for decomp in decomps:
        if decomp.get('factors') is None:
            continue
            
        group = decomp['group']
        factors = decomp['factors']
        weights = decomp.get('weights', np.ones(factors.shape[1]))
        
        # For each component, analyze its structure
        components = []
        for r in range(factors.shape[1]):
            component = factors[:, r]
            
            # Top positive and negative loadings
            pos_idx = np.argsort(component)[-5:][::-1]
            neg_idx = np.argsort(component)[:5]
            
            components.append({
                'component': r,
                'weight': weights[r],
                'top_positive': {
                    'indices': pos_idx.tolist(),
                    'loadings': component[pos_idx].tolist()
                },
                'top_negative': {
                    'indices': neg_idx.tolist(),
                    'loadings': component[neg_idx].tolist()
                },
                'magnitude': np.linalg.norm(component),
                'sparsity': np.sum(np.abs(component) < 0.1) / len(component)  # % near zero
            })
        
        results[group] = {
            'components': components,
            'relative_error': decomp.get('relative_error'),
            'n_samples': decomp.get('n_samples')
        }
    
    return results


def cluster_biomarkers_by_factors(decomps, n_clusters=5):
    """
    Cluster biomarkers based on their loading patterns across groups.
    
    This identifies biomarker modules that behave similarly.
    
    Parameters:
    -----------
    decomps : list of dict
        Decomposition results
    n_clusters : int
        Number of biomarker clusters
        
    Returns:
    --------
    clusters : dict
        Biomarker clusters and their characteristics
    """
    # Concatenate all factors across groups
    all_factors = []
    group_names = []
    
    for decomp in decomps:
        if decomp.get('factors') is not None:
            all_factors.append(decomp['factors'])
            group_names.append(decomp['group'])
    
    if not all_factors:
        return None
    
    # Stack factors: (n_biomarkers, total_components_across_groups)
    stacked = np.hstack(all_factors)
    
    # Cluster biomarkers based on their loading patterns
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(stacked)
    
    # Analyze each cluster
    clusters = {}
    for c in range(n_clusters):
        cluster_biomarkers = np.where(cluster_labels == c)[0]
        cluster_center = kmeans.cluster_centers_[c]
        
        clusters[f'cluster_{c}'] = {
            'biomarker_indices': cluster_biomarkers.tolist(),
            'center': cluster_center.tolist(),
            'size': len(cluster_biomarkers)
        }
    
    return {
        'clusters': clusters,
        'labels': cluster_labels,
        'group_names': group_names
    }


def visualize_group_comparison(analysis_results, biomarker_names=None, save_path=None):
    """
    Create visualizations comparing groups.
    
    Parameters:
    -----------
    analysis_results : dict
        Output from compare_factors_across_groups()
    biomarker_names : list, optional
        Biomarker names for labels
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Similarity matrix heatmap
    sim_matrix = analysis_results['similarity_matrix']
    groups = analysis_results['group_names']
    
    sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=groups, yticklabels=groups, ax=axes[0])
    axes[0].set_title('Group Similarity (Component 1)')
    axes[0].set_xlabel('Group')
    axes[0].set_ylabel('Group')
    
    # 2. Biomarker frequency across groups
    freq = analysis_results['biomarker_frequency']
    if biomarker_names:
        labels = [biomarker_names[idx] if idx < len(biomarker_names) else f'B{idx}' 
                 for idx in sorted(freq.keys())]
    else:
        labels = [f'Biomarker {idx}' for idx in sorted(freq.keys())]
    
    frequencies = [freq[idx] for idx in sorted(freq.keys())]
    axes[1].barh(range(len(labels)), frequencies)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel('Number of Groups')
    axes[1].set_title('Biomarker Frequency Across Groups')
    axes[1].axvline(x=1, color='r', linestyle='--', alpha=0.5, label='Group-specific')
    axes[1].axvline(x=len(groups), color='g', linestyle='--', alpha=0.5, label='All groups')
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_analysis_report(decomps, biomarker_names=None):
    """
    Generate a comprehensive analysis report.
    
    Parameters:
    -----------
    decomps : list of dict
        Decomposition results
    biomarker_names : list, optional
        Biomarker names
        
    Returns:
    --------
    report : dict
        Comprehensive analysis report
    """
    # Compare factors
    comparison = compare_factors_across_groups(decomps)
    
    # Analyze component structure
    component_analysis = analyze_component_structure(decomps, biomarker_names)
    
    # Cluster biomarkers
    clusters = cluster_biomarkers_by_factors(decomps)
    
    # Summary statistics
    errors = [d.get('relative_error') for d in decomps if d.get('relative_error') is not None]
    
    report = {
        'summary': {
            'n_groups': len(decomps),
            'groups_analyzed': [d['group'] for d in decomps if d.get('factors') is not None],
            'mean_reconstruction_error': np.mean(errors) if errors else None,
            'n_biomarkers': decomps[0]['factors'].shape[0] if decomps[0].get('factors') is not None else None,
            'rank': decomps[0]['factors'].shape[1] if decomps[0].get('factors') is not None else None
        },
        'group_comparison': comparison,
        'component_analysis': component_analysis,
        'biomarker_clusters': clusters
    }
    
    return report


def print_analysis_summary(report, biomarker_names=None):
    """Print a human-readable summary of the analysis."""
    print("="*70)
    print("TENSOR DECOMPOSITION ANALYSIS SUMMARY")
    print("="*70)
    
    summary = report['summary']
    print(f"\nGroups analyzed: {summary['n_groups']}")
    print(f"Groups: {', '.join(summary['groups_analyzed'])}")
    print(f"Biomarkers: {summary['n_biomarkers']}")
    print(f"Decomposition rank: {summary['rank']}")
    if summary['mean_reconstruction_error']:
        print(f"Mean reconstruction error: {summary['mean_reconstruction_error']:.4f}")
    
    # Group similarities
    print("\n" + "="*70)
    print("GROUP SIMILARITIES (Component 1)")
    print("="*70)
    sim_matrix = report['group_comparison']['similarity_matrix']
    groups = report['group_comparison']['group_names']
    
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:
                print(f"{group1} vs {group2}: {sim_matrix[i,j]:.4f}")
    
    # Top biomarkers per group
    print("\n" + "="*70)
    print("TOP BIOMARKERS PER GROUP (Component 1)")
    print("="*70)
    top_bio = report['group_comparison']['top_biomarkers']
    
    for group, comps in top_bio.items():
        print(f"\n{group}:")
        if 'component_0' in comps:
            top_idx = comps['component_0']['indices'][:5]
            top_load = comps['component_0']['loadings'][:5]
            for idx, load in zip(top_idx, top_load):
                name = biomarker_names[idx] if biomarker_names and idx < len(biomarker_names) else f"Biomarker {idx}"
                print(f"  {name}: {load:.4f}")
    
    # Shared biomarkers
    print("\n" + "="*70)
    print("SHARED BIOMARKERS (appear in multiple groups)")
    print("="*70)
    shared = report['group_comparison']['shared_biomarkers']
    if shared:
        for idx in shared[:10]:  # Top 10
            name = biomarker_names[idx] if biomarker_names and idx < len(biomarker_names) else f"Biomarker {idx}"
            freq = report['group_comparison']['biomarker_frequency'][idx]
            print(f"  {name}: appears in {freq} groups")
    else:
        print("  No strongly shared biomarkers found")


# Example usage
if __name__ == "__main__":
    # Load results from moments scripts
    from symmetry.moments_3rd_order import main as moments_3rd_main
    
    print("Running 3rd-order moment decomposition...")
    decomps = moments_3rd_main(rank=5)
    
    # Get biomarker names (if available)
    biomarker_names = None  # Replace with actual names if available
    
    # Generate analysis
    report = generate_analysis_report(decomps, biomarker_names)
    
    # Print summary
    print_analysis_summary(report, biomarker_names)
    
    # Visualize (uncomment if matplotlib available)
    # visualize_group_comparison(report['group_comparison'], biomarker_names)

