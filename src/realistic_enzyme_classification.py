#!/usr/bin/env python3
"""
Realistic Enzyme Classification Model
=====================================

This script implements a machine learning model to predict whether a protein
is an enzyme WITHOUT using features that leak target information.

Excluded features: data_richness_score, has_enzyme_keywords
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import os


def create_realistic_features(df):
    """Create features without target leakage."""
    print("=== CREATING REALISTIC FEATURES ===")

    features_df = df.copy()

    # Convert boolean columns to integers
    bool_cols = ['has_keywords', 'has_go_terms',
                 'flag_very_short', 'flag_very_long', 'flag_no_annotation']
    for col in bool_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int)

    # Length-based features
    features_df['log_length'] = np.log1p(features_df['Length'])
    features_df['length_squared'] = features_df['Length'] ** 2
    features_df['length_percentile'] = features_df['Length'].rank(pct=True)

    # Text-based features (general, not enzyme-specific)
    if 'Keywords' in features_df.columns:
        # General keyword count
        features_df['keyword_count'] = features_df['Keywords'].fillna(
            '').str.count(';') + 1
        features_df['keyword_count'] = features_df['keyword_count'].where(
            features_df['Keywords'].notna(), 0)

        # General keyword categories (avoiding enzyme-specific terms)
        structural_keywords = ['3d-structure', 'structure',
                               'domain', 'repeat', 'membrane', 'signal', 'transmembrane']
        metabolic_keywords = ['metabolism', 'biosynthesis',
                              'degradation', 'pathway', 'transport']
        cellular_keywords = ['cell', 'cytoplasm',
                             'nucleus', 'secreted', 'mitochondria', 'ribosome']
        functional_keywords = ['binding', 'activity',
                               'regulation', 'response', 'development']

        features_df['has_structural_keywords'] = features_df['Keywords'].fillna(
            '').str.lower().str.contains('|'.join(structural_keywords), regex=True).astype(int)
        features_df['has_metabolic_keywords'] = features_df['Keywords'].fillna(
            '').str.lower().str.contains('|'.join(metabolic_keywords), regex=True).astype(int)
        features_df['has_cellular_keywords'] = features_df['Keywords'].fillna(
            '').str.lower().str.contains('|'.join(cellular_keywords), regex=True).astype(int)
        features_df['has_functional_keywords'] = features_df['Keywords'].fillna(
            '').str.lower().str.contains('|'.join(functional_keywords), regex=True).astype(int)

    # Gene Ontology features
    if 'Gene_Ontology' in features_df.columns:
        features_df['go_term_count'] = features_df['Gene_Ontology'].fillna(
            '').str.count(';') + 1
        features_df['go_term_count'] = features_df['go_term_count'].where(
            features_df['Gene_Ontology'].notna(), 0)

    # Basic annotation completeness (but not data_richness_score)
    features_df['basic_annotation_score'] = features_df['has_keywords'] + \
        features_df['has_go_terms']

    # Length category encoding
    length_category_map = {'Very_Short': 0, 'Short': 1,
                           'Medium': 2, 'Long': 3, 'Very_Long': 4}
    features_df['length_category_encoded'] = features_df['length_category'].map(
        length_category_map)

    print(f"Created realistic features. Shape: {features_df.shape}")
    return features_df


def train_realistic_models(df):
    """Train models with truly realistic features."""
    print("\n=== TRAINING MODELS WITH REALISTIC FEATURES ===")

    # Exclude all potentially leaky features
    exclude_cols = [
        'Entry', 'EC_number', 'Keywords', 'Gene_Ontology', 'length_category',
        'has_ec_number',  # target
        'data_richness_score',  # leaky feature
        'has_enzyme_keywords'  # leaky feature (if exists)
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Using {len(feature_cols)} realistic features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['has_ec_number'].astype(int)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:")
    print(y.value_counts())
    print(f"Enzyme percentage: {y.mean()*100:.1f}%")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    print("\n--- Logistic Regression Results ---")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores_lr = cross_val_score(
        lr, X_train_scaled, y_train, cv=5, scoring='roc_auc')

    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_roc_auc = roc_auc_score(y_test, y_pred_proba_lr)

    print(
        f"Cross-validation ROC-AUC: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")
    print(f"Test Accuracy: {lr_accuracy:.4f}")
    print(f"Test ROC-AUC: {lr_roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_lr))

    # Feature importance for Logistic Regression
    feature_importance_lr = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(lr.coef_[0])
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (Logistic Regression):")
    print(feature_importance_lr.head(10))

    # Random Forest
    print("\n--- Random Forest Results ---")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)

    # Cross-validation
    cv_scores_rf = cross_val_score(
        rf, X_train, y_train, cv=5, scoring='roc_auc')

    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_roc_auc = roc_auc_score(y_test, y_pred_proba_rf)

    print(
        f"Cross-validation ROC-AUC: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")
    print(f"Test Accuracy: {rf_accuracy:.4f}")
    print(f"Test ROC-AUC: {rf_roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    # Feature importance for Random Forest
    feature_importance_rf = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features (Random Forest):")
    print(feature_importance_rf.head(10))

    results = {
        'logistic_regression': {
            'model': lr,
            'scaler': scaler,
            'accuracy': lr_accuracy,
            'roc_auc': lr_roc_auc,
            'cv_scores': cv_scores_lr,
            'y_pred': y_pred_lr,
            'y_pred_proba': y_pred_proba_lr,
            'feature_importance': feature_importance_lr
        },
        'random_forest': {
            'model': rf,
            'accuracy': rf_accuracy,
            'roc_auc': rf_roc_auc,
            'cv_scores': cv_scores_rf,
            'y_pred': y_pred_rf,
            'y_pred_proba': y_pred_proba_rf,
            'feature_importance': feature_importance_rf
        }
    }

    return results, X_test, y_test, feature_cols


def create_visualizations(results, X_test, y_test, output_dir='outputs/plots'):
    """Create comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== CREATING VISUALIZATIONS ===")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Realistic Enzyme Classification Results',
                 fontsize=16, fontweight='bold')

    # Model performance comparison
    model_names = ['Logistic Regression', 'Random Forest']
    accuracies = [results['logistic_regression']
                  ['accuracy'], results['random_forest']['accuracy']]
    roc_aucs = [results['logistic_regression']
                ['roc_auc'], results['random_forest']['roc_auc']]

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

    # Add value labels on bars
    for i, (acc, auc) in enumerate(zip(accuracies, roc_aucs)):
        axes[0, 0].text(i - width/2, acc + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        axes[0, 0].text(i + width/2, auc + 0.01,
                        f'{auc:.3f}', ha='center', va='bottom')

    # ROC curves
    for name, key in [('Logistic Regression', 'logistic_regression'), ('Random Forest', 'random_forest')]:
        fpr, tpr, _ = roc_curve(y_test, results[key]['y_pred_proba'])
        auc = results[key]['roc_auc']
        axes[0, 1].plot(
            fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5,
                    label='Random Classifier')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Confusion matrices
    for i, (name, key) in enumerate([('Logistic Regression', 'logistic_regression'), ('Random Forest', 'random_forest')]):
        cm = confusion_matrix(y_test, results[key]['y_pred'])
        ax = axes[0, 2] if i == 0 else axes[1, 0]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Enzyme', 'Enzyme'],
                    yticklabels=['Non-Enzyme', 'Enzyme'],
                    ax=ax)
        ax.set_title(f'Confusion Matrix - {name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    # Feature importance plots
    for i, (name, key) in enumerate([('Logistic Regression', 'logistic_regression'), ('Random Forest', 'random_forest')]):
        importance_df = results[key]['feature_importance'].head(10)
        ax = axes[1, 1] if i == 0 else axes[1, 2]

        bars = ax.barh(range(len(importance_df)), importance_df['importance'])
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 10 Features - {name}')
        ax.invert_yaxis()

        # Add value labels
        for j, (idx, row) in enumerate(importance_df.iterrows()):
            ax.text(row['importance'] + max(importance_df['importance']) * 0.01, j,
                    f'{row["importance"]:.3f}', va='center')

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, 'realistic_enzyme_classification.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")

    return fig


def save_results_report(results, output_dir='outputs/reports'):
    """Save a comprehensive results report."""
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, 'enzyme_classification_report.md')

    with open(report_path, 'w') as f:
        f.write("# Enzyme Classification Results\n\n")
        f.write("## Overview\n")
        f.write(
            "This report presents the results of predicting whether proteins are enzymes ")
        f.write("(have EC numbers) using machine learning classification models.\n\n")

        f.write("## Data Leakage Issues Identified and Resolved\n")
        f.write(
            "- **data_richness_score**: Perfect predictor (correlation = 0.88) - EXCLUDED\n")
        f.write(
            "- **has_enzyme_keywords**: Enzyme-specific keyword matching - EXCLUDED\n\n")

        f.write("## Features Used\n")
        f.write("### Numerical Features\n")
        f.write("- Length, log_length, length_squared, length_percentile\n")
        f.write("- keyword_count, go_term_count\n\n")

        f.write("### Boolean Features\n")
        f.write("- has_keywords, has_go_terms\n")
        f.write("- flag_very_short, flag_very_long, flag_no_annotation\n")
        f.write("- has_structural_keywords, has_metabolic_keywords, has_cellular_keywords, has_functional_keywords\n\n")

        f.write("### Derived Features\n")
        f.write("- basic_annotation_score, length_category_encoded\n\n")

        f.write("## Model Performance\n\n")

        for name, key in [('Logistic Regression', 'logistic_regression'), ('Random Forest', 'random_forest')]:
            result = results[key]
            f.write(f"### {name}\n")
            f.write(f"- **Test Accuracy**: {result['accuracy']:.4f}\n")
            f.write(f"- **Test ROC-AUC**: {result['roc_auc']:.4f}\n")
            f.write(
                f"- **Cross-validation ROC-AUC**: {result['cv_scores'].mean():.4f} (+/- {result['cv_scores'].std() * 2:.4f})\n\n")

            f.write(f"#### Top 5 Important Features ({name})\n")
            for idx, row in result['feature_importance'].head(5).iterrows():
                f.write(f"1. {row['feature']}: {row['importance']:.4f}\n")
            f.write("\n")

        f.write("## Conclusions\n")
        best_model = 'logistic_regression' if results['logistic_regression'][
            'roc_auc'] >= results['random_forest']['roc_auc'] else 'random_forest'
        best_result = results[best_model]
        best_name = 'Logistic Regression' if best_model == 'logistic_regression' else 'Random Forest'

        f.write(f"- **Best performing model**: {best_name}\n")
        f.write(f"- **Best test ROC-AUC**: {best_result['roc_auc']:.4f}\n")
        f.write(f"- **Best test accuracy**: {best_result['accuracy']:.4f}\n\n")

        f.write(
            "The models provide realistic performance after removing data leakage issues. ")
        f.write(
            "The classification task demonstrates that protein length, annotation completeness, ")
        f.write(
            "and keyword patterns can help predict enzyme classification, though with moderate accuracy.\n")

    print(f"Results report saved to: {report_path}")


def main():
    """Main function for realistic enzyme classification."""
    print("REALISTIC ENZYME CLASSIFICATION ANALYSIS")
    print("=" * 50)

    # Load data
    df = pd.read_csv('data/proteins_cleaned.tsv', sep='\t')
    print(f"Loaded dataset with {len(df)} proteins")

    # Create realistic features
    df_features = create_realistic_features(df)

    # Train models
    results, X_test, y_test, feature_cols = train_realistic_models(df_features)

    # Create visualizations
    fig = create_visualizations(results, X_test, y_test)

    # Save comprehensive report
    save_results_report(results)

    # Print final summary
    print(f"\n=== FINAL SUMMARY ===")
    best_model = 'logistic_regression' if results['logistic_regression'][
        'roc_auc'] >= results['random_forest']['roc_auc'] else 'random_forest'
    best_result = results[best_model]
    best_name = 'Logistic Regression' if best_model == 'logistic_regression' else 'Random Forest'

    print(f"Best performing model: {best_name}")
    print(f"Test Accuracy: {best_result['accuracy']:.4f}")
    print(f"Test ROC-AUC: {best_result['roc_auc']:.4f}")
    print(
        f"Cross-validation ROC-AUC: {best_result['cv_scores'].mean():.4f} (+/- {best_result['cv_scores'].std() * 2:.4f})")

    return results, df_features


if __name__ == "__main__":
    results, df = main()
