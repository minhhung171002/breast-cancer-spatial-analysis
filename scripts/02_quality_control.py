#!/usr/bin/env python3
"""
Quality Control for Spatial Transcriptomics Data
=================================================
"""

import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configure
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Paths
DATA_DIR = Path("data/raw/breast_cancer_visium")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("results/figures/qc")
IMAGE_DIR = Path("data/images/raw")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_spatial_data():
    """Load Visium spatial transcriptomics data"""
    print("\n" + "=" * 70)
    print("Loading Spatial Transcriptomics Data")
    print("=" * 70)
    
    # Load from matrix market format
    matrix_dir = DATA_DIR / "filtered_feature_bc_matrix"
    
    if not matrix_dir.exists():
        raise FileNotFoundError(
            f"Matrix directory not found: {matrix_dir}\n"
            f"Please run: python scripts/01_download_data.py"
        )
    
    print(f"\nLoading from: {matrix_dir}")
    
    # Load gene expression data
    adata = sc.read_10x_mtx(matrix_dir, var_names='gene_symbols', cache=True)
    
    # Load spatial coordinates
    spatial_dir = DATA_DIR / "spatial"
    if spatial_dir.exists():
        print(f"Loading spatial coordinates from: {spatial_dir}")
        
        # Load tissue positions
        tissue_positions_file = spatial_dir / "tissue_positions_list.csv"
        if not tissue_positions_file.exists():
            tissue_positions_file = spatial_dir / "tissue_positions.csv"
        
        if tissue_positions_file.exists():
            # Read positions
            positions = pd.read_csv(tissue_positions_file, header=None if "list" in tissue_positions_file.name else 0)
            
            # Handle different formats
            if positions.shape[1] >= 5:
                if "list" in tissue_positions_file.name:
                    positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
                
                # Set barcode as index
                positions = positions.set_index('barcode')
                
                # Filter to only barcodes in our data
                positions = positions.loc[adata.obs_names]
                
                # Add spatial coordinates
                adata.obsm['spatial'] = positions[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values
                adata.obs['in_tissue'] = positions['in_tissue'].values
                adata.obs['array_row'] = positions['array_row'].values
                adata.obs['array_col'] = positions['array_col'].values
                
                print("✅ Added spatial coordinates")
        
        # Load scale factors
        scalefactors_file = spatial_dir / "scalefactors_json.json"
        if scalefactors_file.exists():
            with open(scalefactors_file) as f:
                scalefactors = json.load(f)
            
            # Initialize spatial structure properly
            library_id = 'library_1'
            adata.uns['spatial'] = {
                library_id: {
                    'scalefactors': scalefactors,
                    'metadata': {}
                }
            }
            print("✅ Added scale factors")
            
            # Try to load H&E image if available
            image_file = IMAGE_DIR / "breast_cancer_he.tif"
            if image_file.exists():
                try:
                    print(f"Loading H&E image from: {image_file}")
                    from skimage import io
                    img = io.imread(image_file)
                    print(f"  Image shape: {img.shape}")
                    
                    # Resize if needed to match expected dimensions
                    if 'tissue_hires_scalef' in scalefactors:
                        scale = scalefactors['tissue_hires_scalef']
                        from skimage.transform import resize
                        # Estimate target size
                        max_dim = 2000
                        if max(img.shape[:2]) > max_dim:
                            print(f"  Resizing image (original size: {img.shape})...")
                            scale_factor = max_dim / max(img.shape[:2])
                            new_shape = (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor))
                            img = resize(img, new_shape, preserve_range=True, anti_aliasing=True).astype(img.dtype)
                            print(f"  Resized to: {img.shape}")
                    
                    adata.uns['spatial'][library_id]['images'] = {'hires': img}
                    print("✅ Loaded H&E image")
                except Exception as e:
                    print(f"⚠️  Could not load H&E image: {e}")
                    print("   (Plots will work without image)")
    
    print(f"\n✅ Loaded: {adata.n_obs} spots x {adata.n_vars} genes")
    if 'spatial' in adata.obsm:
        print(f"Spatial coordinates: {adata.obsm['spatial'].shape}")
    
    return adata


def calculate_qc_metrics(adata):
    """Calculate QC metrics"""
    print("\nCalculating QC metrics...")
    
    # Identify mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt'],
        percent_top=None,
        log1p=False,
        inplace=True
    )
    
    print("✅ QC metrics calculated")
    print(f"\nQC Summary:")
    print(f"  Mean counts per spot: {adata.obs['total_counts'].mean():.0f}")
    print(f"  Mean genes per spot: {adata.obs['n_genes_by_counts'].mean():.0f}")
    print(f"  Mean MT %: {adata.obs['pct_counts_mt'].mean():.2f}%")
    
    return adata


def plot_qc_spatial(adata):
    """Plot QC metrics on tissue - pure matplotlib version"""
    print("\nGenerating spatial QC plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Check if we have spatial coordinates
    has_spatial = 'spatial' in adata.obsm
    
    if has_spatial:
        # Use pure matplotlib - more reliable than squidpy
        coords = adata.obsm['spatial']
        
        # Total counts
        sc1 = axes[0, 0].scatter(coords[:, 0], coords[:, 1], 
                                  c=adata.obs['total_counts'], 
                                  cmap='viridis', s=5, alpha=0.8)
        axes[0, 0].set_title('Total UMI Counts', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X coordinate')
        axes[0, 0].set_ylabel('Y coordinate')
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[0, 0], label='Counts')
        
        # Number of genes
        sc2 = axes[0, 1].scatter(coords[:, 0], coords[:, 1], 
                                  c=adata.obs['n_genes_by_counts'], 
                                  cmap='viridis', s=5, alpha=0.8)
        axes[0, 1].set_title('Number of Genes', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('X coordinate')
        axes[0, 1].set_ylabel('Y coordinate')
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[0, 1], label='Genes')
        
        # MT percentage
        sc3 = axes[1, 0].scatter(coords[:, 0], coords[:, 1], 
                                  c=adata.obs['pct_counts_mt'], 
                                  cmap='RdYlBu_r', s=5, alpha=0.8)
        axes[1, 0].set_title('% Mitochondrial', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('X coordinate')
        axes[1, 0].set_ylabel('Y coordinate')
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_aspect('equal')
        plt.colorbar(sc3, ax=axes[1, 0], label='MT %')
    else:
        print("⚠️  No spatial coordinates found, creating basic plots...")
        
        # Create alternative plots without spatial
        axes[0, 0].scatter(range(len(adata)), adata.obs['total_counts'], alpha=0.5, s=2)
        axes[0, 0].set_xlabel('Spot Index')
        axes[0, 0].set_ylabel('Total Counts')
        axes[0, 0].set_title('Total UMI Counts', fontsize=12, fontweight='bold')
        
        axes[0, 1].scatter(range(len(adata)), adata.obs['n_genes_by_counts'], alpha=0.5, s=2)
        axes[0, 1].set_xlabel('Spot Index')
        axes[0, 1].set_ylabel('Number of Genes')
        axes[0, 1].set_title('Number of Genes', fontsize=12, fontweight='bold')
        
        axes[1, 0].scatter(range(len(adata)), adata.obs['pct_counts_mt'], alpha=0.5, s=2)
        axes[1, 0].set_xlabel('Spot Index')
        axes[1, 0].set_ylabel('% Mitochondrial')
        axes[1, 0].set_title('% Mitochondrial', fontsize=12, fontweight='bold')
    
    # Distribution histogram (always create this)
    axes[1, 1].hist(adata.obs['total_counts'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Total Counts')
    axes[1, 1].set_ylabel('Number of Spots')
    axes[1, 1].set_title('UMI Count Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qc_spatial.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {FIGURES_DIR / 'qc_spatial.png'}")


def filter_data(adata):
    """Filter spots based on QC metrics"""
    print("\nFiltering data...")
    print(f"Before: {adata.n_obs} spots, {adata.n_vars} genes")
    
    # Filter spots
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, min_counts=500)
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=3)
    
    print(f"After: {adata.n_obs} spots, {adata.n_vars} genes")
    
    return adata


def main():
    print("=" * 70)
    print("🔬 Breast Cancer Spatial Transcriptomics - QC")
    print("=" * 70)
    
    # Load data
    adata = load_spatial_data()
    
    # QC metrics
    adata = calculate_qc_metrics(adata)
    
    # Plot QC
    plot_qc_spatial(adata)
    
    # Filter
    adata = filter_data(adata)
    
    # Save
    output_file = PROCESSED_DIR / 'adata_qc.h5ad'
    adata.write(output_file)
    print(f"\n✅ Saved: {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ Quality Control Complete!")
    print("=" * 70)
    print(f"\nFiltered data: {adata.n_obs} spots x {adata.n_vars} genes")
    print("\nNext step:")
    print("  python scripts/03_preprocessing.py")


if __name__ == "__main__":
    main()