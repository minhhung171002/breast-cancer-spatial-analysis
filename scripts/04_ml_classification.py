#!/usr/bin/env python3
"""
Machine Learning for Cell Type Classification
==============================================
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib

# Paths
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/trained")
RESULTS_DIR = Path("results/ml_results")
FIGURES_DIR = RESULTS_DIR / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def prepare_ml_data(adata, n_genes=2000):
    """Prepare data for ML"""
    print("\n" + "=" * 70)
    print("Preparing ML Data")
    print("=" * 70)
    
    # Get highly variable genes
    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    
    # Features
    X = adata_hvg.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Labels (use leiden clusters) - convert to integers
    y = adata.obs['leiden'].astype(str).astype(int).values
    
    gene_names = adata_hvg.var_names.tolist()
    
    print(f"✅ Prepared: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    
    return X, y, gene_names


def train_models(X_train, y_train, X_test, y_test):
    """Train ML models"""
    print("\n" + "=" * 70)
    print("Training Models")
    print("=" * 70)
    
    models = {}
    results = {}
    
    # Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"✅ RF Accuracy: {acc_rf:.4f}")
    
    models['Random Forest'] = rf
    results['Random Forest'] = {'accuracy': acc_rf, 'y_pred': y_pred_rf}
    
    # XGBoost
    print("\n🚀 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"✅ XGB Accuracy: {acc_xgb:.4f}")
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {'accuracy': acc_xgb, 'y_pred': y_pred_xgb}
    
    return models, results


def plot_confusion_matrix(y_test, y_pred, model_name, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, gene_names, top_n=20):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [gene_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Gene', fontsize=12)
        plt.title(f'Top {top_n} Important Genes', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return [gene_names[i] for i in indices]
    return None


def plot_model_comparison(results_df):
    """Plot model comparison"""
    plt.figure(figsize=(8, 6))
    bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=['steelblue', 'coral'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("🤖 Machine Learning Classification")
    print("=" * 70)
    
    # Load processed data
    adata = sc.read_h5ad(PROCESSED_DIR / 'adata_processed.h5ad')
    print(f"Loaded: {adata.n_obs} spots x {adata.n_vars} genes")
    
    # Prepare data
    X, y, gene_names = prepare_ml_data(adata)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Train models
    models, results = train_models(X_train, y_train, X_test, y_test)
    
    # Plot confusion matrices
    class_names = np.unique(y)
    for name, res in results.items():
        plot_confusion_matrix(y_test, res['y_pred'], name, class_names)
        print(f"✅ Saved confusion matrix for {name}")
    
    # Feature importance
    top_genes = plot_feature_importance(models['Random Forest'], gene_names, top_n=20)
    if top_genes:
        print(f"✅ Saved feature importance plot")
    
    # Save models
    for name, model in models.items():
        model_file = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_file)
        print(f"✅ Saved model: {model_file}")
    
    # Save results
    results_df = pd.DataFrame([
        {'Model': name, 'Accuracy': res['accuracy']}
        for name, res in results.items()
    ])
    results_df.to_csv(RESULTS_DIR / 'model_performance.csv', index=False)
    
    # Plot model comparison
    plot_model_comparison(results_df)
    print(f"✅ Saved model comparison plot")
    
    print("\n" + "=" * 70)
    print("✅ ML Classification Complete!")
    print("=" * 70)
    print("\nModel Performance:")
    print(results_df.to_string(index=False))
    
    if top_genes:
        print(f"\nTop 5 Predictive Genes:")
        for i, gene in enumerate(top_genes[:5], 1):
            print(f"  {i}. {gene}")
    
    print(f"\nGenerated plots:")
    print(f"  - {FIGURES_DIR / 'confusion_matrix_random_forest.png'}")
    print(f"  - {FIGURES_DIR / 'confusion_matrix_xgboost.png'}")
    print(f"  - {FIGURES_DIR / 'feature_importance.png'}")
    print(f"  - {FIGURES_DIR / 'model_comparison.png'}")
    
    print("\nNext step:")
    print("  python scripts/05_image_analysis.py")


if __name__ == "__main__":
    main()