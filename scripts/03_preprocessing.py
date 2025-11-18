#!/usr/bin/env python3
"""
Preprocessing and Dimensionality Reduction
===========================================
"""

import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Configure
sc.settings.verbosity = 3

# Paths
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("results/figures/preprocessing")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def normalize_and_process(adata):
    """Normalize and process data"""
    print("\n" + "=" * 70)
    print("Normalizing and Processing")
    print("=" * 70)
    
    # Store raw counts
    adata.layers['counts'] = adata.X.copy()
    
    # Normalize
    print("\n1. Normalizing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['log_normalized'] = adata.X.copy()
    
    # Highly variable genes
    print("\n2. Finding highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3', layer='counts')
    print(f"Found {adata.var['highly_variable'].sum()} HVGs")
    
    # Scale
    print("\n3. Scaling...")
    sc.pp.scale(adata, max_value=10)
    
    return adata


def dimensionality_reduction(adata):
    """PCA and UMAP"""
    print("\n4. Running PCA...")
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    
    # Plot variance
    sc.pl.pca_variance_ratio(adata, n_pcs=50, show=False)
    plt.savefig(FIGURES_DIR / 'pca_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n5. Computing neighbors...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    
    print("\n6. Running UMAP...")
    sc.tl.umap(adata)
    
    return adata


def clustering(adata):
    """Cluster analysis"""
    print("\n7. Clustering...")
    sc.tl.leiden(adata, resolution=0.5)
    
    # Create plots using matplotlib instead of squidpy
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Check if we have spatial coordinates
    has_spatial = 'spatial' in adata.obsm
    
    if has_spatial:
        # Plot clusters on spatial coordinates
        coords = adata.obsm['spatial']
        
        # Get cluster colors
        cluster_labels = adata.obs['leiden'].astype('category')
        n_clusters = len(cluster_labels.cat.categories)
        
        # Use a color palette
        from matplotlib import cm
        colors = cm.tab20(np.linspace(0, 1, n_clusters))
        color_dict = dict(zip(cluster_labels.cat.categories, colors))
        
        # Map clusters to colors
        point_colors = [color_dict[label] for label in cluster_labels]
        
        # Spatial scatter
        axes[0].scatter(coords[:, 0], coords[:, 1], 
                       c=point_colors, s=5, alpha=0.8)
        axes[0].set_title('Clusters on Tissue', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')
        axes[0].invert_yaxis()
        axes[0].set_aspect('equal')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_dict[cat], label=cat) 
                          for cat in cluster_labels.cat.categories]
        axes[0].legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1, 0.5), title='Cluster')
    else:
        axes[0].text(0.5, 0.5, 'No spatial coordinates available', 
                    ha='center', va='center', fontsize=12)
        axes[0].set_title('Clusters on Tissue', fontsize=12, fontweight='bold')
    
    # UMAP plot
    sc.pl.umap(adata, color='leiden', ax=axes[1], show=False, legend_loc='on data')
    axes[1].set_title('Clusters (UMAP)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Found {adata.obs['leiden'].nunique()} clusters")
    
    return adata


def main():
    print("=" * 70)
    print("🧬 Preprocessing and Clustering")
    print("=" * 70)
    
    # Load QC'd data
    adata = sc.read_h5ad(PROCESSED_DIR / 'adata_qc.h5ad')
    print(f"\nLoaded: {adata.n_obs} spots x {adata.n_vars} genes")
    
    # Process
    adata = normalize_and_process(adata)
    adata = dimensionality_reduction(adata)
    adata = clustering(adata)
    
    # Save
    output_file = PROCESSED_DIR / 'adata_processed.h5ad'
    adata.write(output_file)
    print(f"\n✅ Saved: {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ Preprocessing Complete!")
    print("=" * 70)
    print(f"\nProcessed data: {adata.n_obs} spots x {adata.n_vars} genes")
    print(f"Generated plots:")
    print(f"  - {FIGURES_DIR / 'pca_variance.png'}")
    print(f"  - {FIGURES_DIR / 'clusters.png'}")
    print("\nNext step:")
    print("  python scripts/04_ml_classification.py")


if __name__ == "__main__":
    main()