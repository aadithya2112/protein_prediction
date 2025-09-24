#!/usr/bin/env python3
"""
Enzyme Classification Model
==========================

This script implements a machine learning model to predict whether a protein
is an enzyme (has EC number) based on available protein features.

Target: Binary classification of has_ec_number (True/False)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def load_and_explore_data(file_path):
    """Load the cleaned protein data and perform initial exploration."""
    print("Loading protein data...")
    df = pd.read_csv(file_path, sep='\t')

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Basic info about the dataset
    print("\n=== DATASET OVERVIEW ===")
    print(df.info())

    # Target variable distribution
    print("\n=== TARGET VARIABLE (has_ec_number) DISTRIBUTION ===")
    target_dist = df['has_ec_number'].value_counts()
    print(target_dist)
    print(f"Enzyme percentage: {target_dist[True] / len(df) * 100:.1f}%")

    # Check for missing values
    print("\n=== MISSING VALUES ===")
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0])

    # Basic statistics
    print("\n=== NUMERICAL FEATURES SUMMARY ===")
    numerical_cols = ['Length', 'data_richness_score']
    if all(col in df.columns for col in numerical_cols):
        print(df[numerical_cols].describe())

    return df


def create_features(df):
    """Create and engineer features for enzyme classification."""
    print("\n=== FEATURE ENGINEERING ===")

    # Make a copy to avoid modifying original data
    features_df = df.copy()

    # Convert boolean columns to integers
    bool_cols = ['has_ec_number', 'has_keywords', 'has_go_terms',
                 'flag_very_short', 'flag_very_long', 'flag_no_annotation']

    for col in bool_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int)

    # Create length-based features
    if 'Length' in features_df.columns:
        features_df['log_length'] = np.log1p(features_df['Length'])
        features_df['length_squared'] = features_df['Length'] ** 2

        # Length percentile features
        features_df['length_percentile'] = features_df['Length'].rank(pct=True)

    # Create text-based features from Keywords
    if 'Keywords' in features_df.columns:
        # Count keywords (simple approach)
        features_df['keyword_count'] = features_df['Keywords'].fillna(
            '').str.count(';') + 1
        features_df['keyword_count'] = features_df['keyword_count'].where(
            features_df['Keywords'].notna(), 0)

        # Check for enzyme-related keywords
        enzyme_keywords = ['enzyme', 'kinase', 'phosphatase', 'dehydrogenase',
                           'transferase', 'hydrolase', 'oxidase', 'reductase']
        features_df['has_enzyme_keywords'] = features_df['Keywords'].fillna('').str.lower().str.contains(
            '|'.join(enzyme_keywords), regex=True).astype(int)

    # Create annotation completeness score
    annotation_cols = ['has_keywords', 'has_go_terms']
    if all(col in features_df.columns for col in annotation_cols):
        features_df['annotation_completeness'] = features_df[annotation_cols].sum(
            axis=1)

    # Encode length category if exists
    if 'length_category' in features_df.columns:
        le = LabelEncoder()
        features_df['length_category_encoded'] = le.fit_transform(
            features_df['length_category'])

    print(f"Features created. New shape: {features_df.shape}")
    return features_df


def prepare_model_data(df, target_col='has_ec_number'):
    """Prepare features and target for machine learning."""
    print(f"\n=== PREPARING DATA FOR MODELING ===")

    # Define feature columns (exclude target and identifier columns)
    exclude_cols = ['Entry', 'EC_number', 'Keywords', 'Gene_Ontology',
                    'length_category', target_col]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Using {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    # Prepare feature matrix and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle any remaining missing values
    X = X.fillna(0)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:")
    print(y.value_counts())

    return X, y, feature_cols


def train_and_evaluate_models(X, y, feature_names):
    """Train and evaluate different models for enzyme classification."""
    print(f"\n=== MODEL TRAINING AND EVALUATION ===")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models to try
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            X_test_model = X_test_scaled
        else:
            model.fit(X_train, y_train)
            X_test_model = X_test

        # Cross-validation
        if name == 'Logistic Regression':
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        else:
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring='roc_auc')

        # Predictions
        y_pred = model.predict(X_test_model)
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results[name] = {
            'model': model,
            'scaler': scaler if name == 'Logistic Regression' else None,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_scores': cv_scores,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }

        print(
            f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test ROC-AUC: {roc_auc:.4f}")

        # Classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))

    return results, X_train, X_test, y_train, y_test, feature_names


def plot_results(results, output_dir='outputs/plots'):
    """Create visualizations of model results."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== CREATING VISUALIZATIONS ===")

    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enzyme Classification Model Results',
                 fontsize=16, fontweight='bold')

    # Model comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    roc_aucs = [results[name]['roc_auc'] for name in model_names]

    # Bar plot of metrics
    x = np.arange(len(model_names))
    width = 0.35

    axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    axes[0, 0].bar(x + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])

    # ROC curves
    for name in model_names:
        fpr, tpr, _ = roc_curve(
            results[name]['y_test'], results[name]['y_pred_proba'])
        auc = results[name]['roc_auc']
        axes[0, 1].plot(
            fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Confusion matrices for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    best_result = results[best_model_name]

    # Confusion matrix
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Enzyme', 'Enzyme'],
                yticklabels=['Non-Enzyme', 'Enzyme'],
                ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')

    # Feature importance (for Random Forest)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        # We'll need feature names for this
        axes[1, 1].set_title('Feature Importance (Random Forest)')
        axes[1, 1].text(0.5, 0.5, 'Feature importance plot\nwill be generated\nwith feature names',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].axis('off')

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, 'enzyme_classification_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Results plot saved to: {plot_path}")

    return fig


def main():
    """Main function to run the enzyme classification analysis."""
    print("ENZYME CLASSIFICATION ANALYSIS")
    print("=" * 50)

    # Load and explore data
    data_path = 'data/proteins_cleaned.tsv'
    df = load_and_explore_data(data_path)

    # Create features
    df_features = create_features(df)

    # Prepare data for modeling
    X, y, feature_names = prepare_model_data(df_features)

    # Train and evaluate models
    results, X_train, X_test, y_train, y_test, feature_names = train_and_evaluate_models(
        X, y, feature_names)

    # Create visualizations
    fig = plot_results(results)

    # Print summary
    print(f"\n=== FINAL SUMMARY ===")
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    best_result = results[best_model_name]

    print(f"Best performing model: {best_model_name}")
    print(f"Test Accuracy: {best_result['accuracy']:.4f}")
    print(f"Test ROC-AUC: {best_result['roc_auc']:.4f}")
    print(
        f"Cross-validation ROC-AUC: {best_result['cv_scores'].mean():.4f} (+/- {best_result['cv_scores'].std() * 2:.4f})")

    return results, df_features


if __name__ == "__main__":
    results, df = main()
