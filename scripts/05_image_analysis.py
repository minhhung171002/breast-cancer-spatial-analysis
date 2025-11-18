#!/usr/bin/env python3
"""
H&E Image Analysis
==================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, color, filters, measure, morphology

# Paths
IMAGE_DIR = Path("data/images/raw")
FIGURES_DIR = Path("results/figures/image_analysis")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_he_image():
    """Load H&E image"""
    print("\n" + "=" * 70)
    print("Loading H&E Image")
    print("=" * 70)
    
    image_path = IMAGE_DIR / "breast_cancer_he.tif"
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    img = io.imread(image_path)
    print(f"✅ Loaded image: {img.shape}")
    
    return img


def preprocess_image(img):
    """Preprocess H&E image"""
    print("\nPreprocessing image...")
    
    # Resize if too large (for faster processing)
    max_size = 2000
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        new_shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_shape)
        print(f"Resized to: {img.shape}")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img
    
    # Normalize
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    
    return img, gray


def segment_nuclei(gray):
    """Segment cell nuclei"""
    print("\nSegmenting nuclei...")
    
    # Threshold
    threshold = filters.threshold_otsu(gray)
    nuclei_mask = gray < threshold
    
    # Clean up
    nuclei_mask = morphology.remove_small_objects(nuclei_mask, min_size=100)
    nuclei_mask = morphology.remove_small_holes(nuclei_mask, area_threshold=100)
    
    # Label regions
    labeled = measure.label(nuclei_mask)
    properties = measure.regionprops(labeled)
    
    print(f"✅ Detected {len(properties)} nuclei")
    
    return nuclei_mask, labeled, properties


def visualize_results(img, gray, nuclei_mask, labeled):
    """Visualize segmentation results"""
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original H&E Image')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # Mask
    axes[1, 0].imshow(nuclei_mask, cmap='gray')
    axes[1, 0].set_title('Nuclei Mask')
    axes[1, 0].axis('off')
    
    # Labeled
    axes[1, 1].imshow(labeled, cmap='nipy_spectral')
    axes[1, 1].set_title('Segmented Nuclei')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'image_segmentation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {FIGURES_DIR / 'image_segmentation.png'}")


def extract_features(properties):
    """Extract morphological features"""
    print("\nExtracting morphological features...")
    
    features = []
    for prop in properties:
        features.append({
            'area': prop.area,
            'perimeter': prop.perimeter,
            'eccentricity': prop.eccentricity,
            'solidity': prop.solidity,
        })
    
    import pandas as pd
    features_df = pd.DataFrame(features)
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(['area', 'perimeter', 'eccentricity', 'solidity']):
        axes[idx].hist(features_df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(col.title())
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{col.title()} Distribution')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved feature distributions")
    
    return features_df


def main():
    print("=" * 70)
    print("🖼️  H&E Image Analysis")
    print("=" * 70)
    
    # Load image
    img = load_he_image()
    if img is None:
        return
    
    # Preprocess
    img, gray = preprocess_image(img)
    
    # Segment nuclei
    nuclei_mask, labeled, properties = segment_nuclei(gray)
    
    # Visualize
    visualize_results(img, gray, nuclei_mask, labeled)
    
    # Extract features
    features_df = extract_features(properties)
    
    # Save features
    Path("results/ml_results").mkdir(parents=True, exist_ok=True)
    features_df.to_csv(Path("results/ml_results") / 'cell_morphology_features.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✅ Image Analysis Complete!")
    print("=" * 70)
    print(f"\nDetected {len(properties)} cells")
    print("\nFeature Summary:")
    print(features_df.describe())


if __name__ == "__main__":
    main()