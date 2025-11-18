#!/usr/bin/env python3
"""
Download Breast Cancer Data
============================
Downloads 10x Genomics Visium breast cancer data with H&E imaging.
"""
import os
import sys
import tarfile
from pathlib import Path
from tqdm import tqdm
import requests

# Setup paths
DATA_DIR = Path("data/raw/breast_cancer_visium")
IMAGE_DIR = Path("data/images/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, output_path):
    """Download file with progress bar using requests"""
    print(f"\n📥 Downloading: {output_path.name}")
    print(f"From: {url}")
    
    # Add headers to avoid 403 errors
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Stream download with progress bar
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, miniters=1) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Downloaded successfully!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading: {e}")
        if output_path.exists():
            output_path.unlink()  # Remove partial download
        return False

def main():
    print("=" * 70)
    print("🎗️  Breast Cancer Data Downloader")
    print("=" * 70)
    print("\nDataset: 10x Genomics Visium Breast Cancer")
    print("Type: Spatial Transcriptomics with H&E Imaging")
    print("Size: ~100 MB")
    print()
    
    # Check if requests is installed
    try:
        import requests
    except ImportError:
        print("❌ Error: 'requests' library not found")
        print("Installing requests...")
        os.system(f"{sys.executable} -m pip install requests")
        import requests
    
    # URLs for breast cancer spatial data
    urls = {
        'matrix': 'https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.tar.gz',
        'spatial': 'https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz',
        'image': 'https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_image.tif'
    }
    
    # Download gene expression matrix
    matrix_file = DATA_DIR / "filtered_feature_bc_matrix.tar.gz"
    if not matrix_file.exists():
        if download_file(urls['matrix'], matrix_file):
            print("Extracting gene expression data...")
            with tarfile.open(matrix_file, 'r:gz') as tar:
                tar.extractall(DATA_DIR)
            print("✅ Extracted")
        else:
            print("❌ Failed to download gene expression matrix")
            sys.exit(1)
    else:
        print("\n✓ Gene expression matrix already exists")
    
    # Download spatial coordinates
    spatial_file = DATA_DIR / "spatial.tar.gz"
    if not spatial_file.exists():
        if download_file(urls['spatial'], spatial_file):
            print("Extracting spatial coordinates...")
            with tarfile.open(spatial_file, 'r:gz') as tar:
                tar.extractall(DATA_DIR)
            print("✅ Extracted")
        else:
            print("❌ Failed to download spatial coordinates")
            sys.exit(1)
    else:
        print("\n✓ Spatial coordinates already exist")
    
    # Download H&E image
    image_file = IMAGE_DIR / "breast_cancer_he.tif"
    if not image_file.exists():
        if not download_file(urls['image'], image_file):
            print("❌ Failed to download H&E image")
            sys.exit(1)
    else:
        print("\n✓ H&E image already exists")
    
    print("\n" + "=" * 70)
    print("✅ All Data Downloaded Successfully!")
    print("=" * 70)
    print(f"\nData location: {DATA_DIR}")
    print(f"Image location: {image_file}")
    print("\nNext steps:")
    print("  python scripts/02_quality_control.py")
    print("=" * 70)

if __name__ == "__main__":
    main()