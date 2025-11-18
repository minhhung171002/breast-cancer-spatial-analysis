
# 🎗️ Breast Cancer Spatial Transcriptomics Analysis

Complete machine learning and image analysis pipeline for spatial transcriptomics data.

## Overview

This project analyzes breast cancer tissue using 10x Genomics Visium spatial transcriptomics data, combining:
- Gene expression profiling
- Spatial clustering analysis  
- Machine learning classification (88.7% RF, 90.96% XGBoost)
- Histological image analysis

## Key Results

- **10 distinct tissue regions** identified
- **90.96% classification accuracy** with XGBoost
- **Top 5 marker genes:** CRISP3, IGFBP5, MGP, SLITRK6, CPB1
- **352 cells** segmented from H&E images

## Project Structure
```
breast_cancer_ml_project/
├── scripts/              # Analysis pipeline (5 scripts)
│   ├── 01_download_data.py
│   ├── 02_quality_control.py
│   ├── 03_preprocessing.py
│   ├── 04_ml_classification.py
│   └── 05_image_analysis.py
├── data/
│   ├── raw/              # Raw data from 10x Genomics
│   └── processed/        # Processed AnnData objects
├── results/
│   ├── figures/          # All generated plots
│   └── ml_results/       # Model performance metrics
├── models/
│   └── trained/          # Saved ML models
└── requirements.txt      # Python dependencies
```

##Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/minhhung171002/breast-cancer-spatial-analysis.git
cd breast-cancer-spatial-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install scikit-misc  # Additional requirement
```

### Run Analysis
```bash
# Run individual steps
python scripts/01_download_data.py      # ~5 min
python scripts/02_quality_control.py    # ~2 min
python scripts/03_preprocessing.py      # ~3 min
python scripts/04_ml_classification.py  # ~5 min
python scripts/05_image_analysis.py     # ~3 min
```

## 📊 Pipeline Stages

### 1. Data Download (5 min)
Downloads spatial transcriptomics data from 10x Genomics repository.

### 2. Quality Control (2 min)
- Filters low-quality spots and genes
- Calculates QC metrics (UMI counts, gene counts, mitochondrial %)
- Generates spatial QC visualizations

**Results:** 3,813 spots × 20,687 genes (filtered from 33,538)

### 3. Preprocessing (3 min)
- Normalizes gene expression
- Identifies 3,000 highly variable genes
- Performs PCA and UMAP
- Leiden clustering

**Results:** 10 distinct spatial clusters identified

### 4. Machine Learning (5 min)
- Trains Random Forest and XGBoost classifiers
- Evaluates on test set
- Identifies top marker genes

**Results:**
- Random Forest: 88.73% accuracy
- XGBoost: 90.96% accuracy

### 5. Image Analysis (3 min)
- Segments cell nuclei from H&E image
- Extracts morphological features
- Generates quantitative metrics

**Results:** 352 cells detected and characterized

## 📈 Key Findings

### Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | 88.73% |
| XGBoost | **90.96%** |

### Top 5 Predictive Genes

1. **CRISP3** - Cysteine-rich secretory protein 3
2. **IGFBP5** - Insulin-like growth factor binding protein 5
3. **MGP** - Matrix Gla protein
4. **SLITRK6** - SLIT and NTRK-like family member 6
5. **CPB1** - Carboxypeptidase B1

### Data Quality

- Mean UMI counts per spot: 18,607
- Mean genes per spot: 5,064
- Mitochondrial %: 4.48% (excellent quality)

### Cell Morphology

- Total cells detected: 352
- Mean cell area: 2,432 pixels²
- Mean eccentricity: 0.41 (slightly elongated)

## 🛠️ Requirements

- Python 3.11+
- 16 GB RAM (minimum)
- ~10 GB storage
- macOS, Linux, or Windows

### Key Dependencies
```
scanpy==1.9.6
squidpy==1.3.0
scikit-learn==1.3.2
xgboost==2.0.3
scikit-image==0.22.0
scikit-misc==0.5.2
```

See `requirements.txt` for complete list.

## 📊 Generated Outputs

### Figures (11 total)
- `qc_spatial.png` - Quality control metrics
- `pca_variance.png` - PCA variance explained
- `clusters.png` - Spatial clusters and UMAP
- `confusion_matrix_*.png` - Model performance (2)
- `feature_importance.png` - Top genes
- `model_comparison.png` - Algorithm comparison
- `image_segmentation.png` - Cell segmentation
- `feature_distributions.png` - Morphology stats

### Data Files
- `adata_qc.h5ad` - Quality-controlled data
- `adata_processed.h5ad` - Clustered data
- `model_performance.csv` - ML metrics
- `cell_morphology_features.csv` - Cell measurements

### Models
- `random_forest.pkl` - Trained RF classifier
- `xgboost.pkl` - Trained XGBoost classifier

## 🔬 Methods

### Data Processing
1. **Normalization:** Total count normalization (10,000 counts/spot)
2. **Feature Selection:** Top 3,000 highly variable genes (Seurat v3)
3. **Dimensionality Reduction:** PCA (50 components) → UMAP (2D)
4. **Clustering:** Leiden algorithm (resolution=0.5)

### Machine Learning
- **Train/Test Split:** 80/20 stratified
- **Features:** 3,000 highly variable genes
- **Models:** Random Forest (100 trees), XGBoost (100 rounds)
- **Evaluation:** Accuracy, confusion matrices, feature importance

### Image Analysis
- **Segmentation:** Otsu thresholding + morphological operations
- **Features:** Area, perimeter, eccentricity, solidity
- **Resolution:** 2000×2000 pixels (resized from 24240×24240)


## 📝 Citation

If you use this pipeline, please cite:
```bibtex
@software{breast_cancer_spatial_analysis,
  author = {Le Minh Hung},
  title = {Breast Cancer Spatial Transcriptomics Analysis},
  year = {2025},
  url = {https://github.com/minhhung171002/breast-cancer-spatial-analysis}
}
```

## 📚 References

- **Data Source:** 10x Genomics Visium Breast Cancer Dataset
- **Scanpy:** Wolf et al. (2018) Genome Biology
- **Squidpy:** Palla et al. (2022) Nature Methods
- **XGBoost:** Chen & Guestrin (2016) KDD



## Acknowledgments

- 10x Genomics for public datasets
- Scanpy and Squidpy development teams
- Open-source bioinformatics community

## Contact

For questions or issues:
- GitHub Issues: [Create an issue](https://github.com/minhhung171002/breast-cancer-spatial-analysis/issues)
- Email: minhhung171002@gmail.com
