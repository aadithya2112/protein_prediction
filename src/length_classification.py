#!/usr/bin/env python3
"""
Protein Length Category Classification
======================================

This script implements machine learning models to predict protein length categories
(Very_Short, Short, Medium, Long, Very_Long) based on protein features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import os


def create_features_for_length_prediction(df):
    """Create features for length category prediction."""
    print("=== CREATING FEATURES FOR LENGTH CLASSIFICATION ===")

    features_df = df.copy()

    # Convert boolean columns to integers
    bool_cols = ['has_keywords', 'has_go_terms', 'has_ec_number',
                 'flag_very_short', 'flag_very_long', 'flag_no_annotation']
    for col in bool_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int)

    # Text-based features
    if 'Keywords' in features_df.columns:
        # Keyword count and types
        features_df['keyword_count'] = features_df['Keywords'].fillna(
            '').str.count(';') + 1
        features_df['keyword_count'] = features_df['keyword_count'].where(
            features_df['Keywords'].notna(), 0)

        # Keyword categories
        structural_keywords = ['3d-structure', 'structure',
                               'domain', 'repeat', 'membrane', 'signal']
        metabolic_keywords = ['metabolism',
                              'biosynthesis', 'degradation', 'pathway']
        cellular_keywords = ['cell', 'cytoplasm',
                             'nucleus', 'secreted', 'mitochondria']

        features_df['has_structural_keywords'] = features_df['Keywords'].fillna(
            '').str.lower().str.contains('|'.join(structural_keywords), regex=True).astype(int)
        features_df['has_metabolic_keywords'] = features_df['Keywords'].fillna(
            '').str.lower().str.contains('|'.join(metabolic_keywords), regex=True).astype(int)
        features_df['has_cellular_keywords'] = features_df['Keywords'].fillna(
            '').str.lower().str.contains('|'.join(cellular_keywords), regex=True).astype(int)

    # Gene Ontology features
    if 'Gene_Ontology' in features_df.columns:
        features_df['go_term_count'] = features_df['Gene_Ontology'].fillna(
            '').str.count(';') + 1
        features_df['go_term_count'] = features_df['go_term_count'].where(
            features_df['Gene_Ontology'].notna(), 0)

    # Annotation completeness
    features_df['annotation_score'] = (features_df['has_keywords'] +
                                       features_df['has_go_terms'] +
                                       features_df['has_ec_number'])

    print(f"Created features. Shape: {features_df.shape}")
    return features_df


def train_length_classification_models(df):
    """Train models to predict protein length categories."""
    print("\n=== TRAINING LENGTH CLASSIFICATION MODELS ===")

    # Exclude Length-based features to make it realistic
    exclude_cols = ['Entry', 'EC_number', 'Keywords', 'Gene_Ontology',
                    # Exclude direct length features
                    'Length', 'log_length', 'length_squared', 'length_percentile',
                    'length_category', 'data_richness_score']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Using {len(feature_cols)} features for length prediction:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['length_category']

    # Check class distribution
    print(f"\nLength category distribution:")
    print(y.value_counts().sort_index())

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(
        f"\nEncoded classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Logistic Regression (multiclass)
    print("\n--- Logistic Regression Results ---")
    lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
    lr.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores_lr = cross_val_score(
        lr, X_train_scaled, y_train, cv=5, scoring='accuracy')

    y_pred_lr = lr.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')

    print(
        f"Cross-validation Accuracy: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")
    print(f"Test Accuracy: {lr_accuracy:.4f}")
    print(f"Test F1-score (weighted): {lr_f1:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

    # Random Forest
    print("\n--- Random Forest Results ---")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)

    # Cross-validation
    cv_scores_rf = cross_val_score(
        rf, X_train, y_train, cv=5, scoring='accuracy')

    y_pred_rf = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

    print(
        f"Cross-validation Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")
    print(f"Test Accuracy: {rf_accuracy:.4f}")
    print(f"Test F1-score (weighted): {rf_f1:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

    # Feature importance
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
            'f1_score': lr_f1,
            'cv_scores': cv_scores_lr,
            'y_pred': y_pred_lr
        },
        'random_forest': {
            'model': rf,
            'accuracy': rf_accuracy,
            'f1_score': rf_f1,
            'cv_scores': cv_scores_rf,
            'y_pred': y_pred_rf,
            'feature_importance': feature_importance_rf
        },
        'label_encoder': le,
        'y_test': y_test
    }

    return results, X_test, feature_cols


def create_length_visualizations(results, X_test, output_dir='outputs/plots'):
    """Create visualizations for length classification."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== CREATING LENGTH CLASSIFICATION VISUALIZATIONS ===")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Protein Length Category Classification Results',
                 fontsize=16, fontweight='bold')

    # Model performance comparison
    model_names = ['Logistic Regression', 'Random Forest']
    accuracies = [results['logistic_regression']
                  ['accuracy'], results['random_forest']['accuracy']]
    f1_scores = [results['logistic_regression']
                 ['f1_score'], results['random_forest']['f1_score']]

    x = np.arange(len(model_names))
    width = 0.35

    axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    axes[0, 0].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])

    # Add value labels
    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
        axes[0, 0].text(i - width/2, acc + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        axes[0, 0].text(i + width/2, f1 + 0.01,
                        f'{f1:.3f}', ha='center', va='bottom')

    # Confusion matrices
    le = results['label_encoder']

    for i, (name, key) in enumerate([('Random Forest', 'random_forest'), ('Logistic Regression', 'logistic_regression')]):
        cm = confusion_matrix(results['y_test'], results[key]['y_pred'])
        ax = axes[0, 1] if i == 0 else axes[1, 0]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_,
                    yticklabels=le.classes_,
                    ax=ax)
        ax.set_title(f'Confusion Matrix - {name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    # Feature importance (Random Forest)
    importance_df = results['random_forest']['feature_importance'].head(10)
    axes[1, 1].barh(range(len(importance_df)), importance_df['importance'])
    axes[1, 1].set_yticks(range(len(importance_df)))
    axes[1, 1].set_yticklabels(importance_df['feature'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Features - Random Forest')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, 'length_classification_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Length classification visualization saved to: {plot_path}")

    return fig


def save_length_classification_report(results, output_dir='outputs/reports'):
    """Save length classification report."""
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, 'length_classification_report.md')

    with open(report_path, 'w') as f:
        f.write("# Protein Length Category Classification Results\n\n")
        f.write("## Overview\n")
        f.write(
            "This report presents the results of predicting protein length categories ")
        f.write(
            "(Very_Short, Short, Medium, Long, Very_Long) using machine learning models.\n\n")

        f.write("## Challenge\n")
        f.write("Predicting length categories WITHOUT using direct length features ")
        f.write("(Length, log_length, etc.) to test if other protein characteristics ")
        f.write("can indicate protein size.\n\n")

        f.write("## Features Used\n")
        f.write("### Boolean Features\n")
        f.write("- has_keywords, has_go_terms, has_ec_number\n")
        f.write("- flag_very_short, flag_very_long, flag_no_annotation\n")
        f.write(
            "- has_structural_keywords, has_metabolic_keywords, has_cellular_keywords\n\n")

        f.write("### Numerical Features\n")
        f.write("- keyword_count, go_term_count\n")
        f.write("- annotation_score\n\n")

        f.write("## Model Performance\n\n")

        for name, key in [('Logistic Regression', 'logistic_regression'), ('Random Forest', 'random_forest')]:
            result = results[key]
            f.write(f"### {name}\n")
            f.write(f"- **Test Accuracy**: {result['accuracy']:.4f}\n")
            f.write(
                f"- **Test F1-Score (weighted)**: {result['f1_score']:.4f}\n")
            f.write(
                f"- **Cross-validation Accuracy**: {result['cv_scores'].mean():.4f} (+/- {result['cv_scores'].std() * 2:.4f})\n\n")

        if 'feature_importance' in results['random_forest']:
            f.write("#### Top 5 Important Features (Random Forest)\n")
            for idx, row in results['random_forest']['feature_importance'].head(5).iterrows():
                f.write(f"1. {row['feature']}: {row['importance']:.4f}\n")
            f.write("\n")

        f.write("## Conclusions\n")
        best_model = 'random_forest' if results['random_forest']['accuracy'] >= results[
            'logistic_regression']['accuracy'] else 'logistic_regression'
        best_result = results[best_model]
        best_name = 'Random Forest' if best_model == 'random_forest' else 'Logistic Regression'

        f.write(f"- **Best performing model**: {best_name}\n")
        f.write(f"- **Best test accuracy**: {best_result['accuracy']:.4f}\n")
        f.write(f"- **Best test F1-score**: {best_result['f1_score']:.4f}\n\n")

        f.write(
            "The models show that predicting protein length categories without direct ")
        f.write(
            "length information is challenging but possible using annotation patterns ")
        f.write("and functional characteristics.\n")

    print(f"Length classification report saved to: {report_path}")


def main():
    """Main function for length classification."""
    print("PROTEIN LENGTH CATEGORY CLASSIFICATION")
    print("=" * 45)

    # Load data
    df = pd.read_csv('data/proteins_cleaned.tsv', sep='\t')
    print(f"Loaded dataset with {len(df)} proteins")

    # Create features
    df_features = create_features_for_length_prediction(df)

    # Train models
    results, X_test, feature_cols = train_length_classification_models(
        df_features)

    # Create visualizations
    fig = create_length_visualizations(results, X_test)

    # Save report
    save_length_classification_report(results)

    # Print summary
    print(f"\n=== FINAL SUMMARY ===")
    best_model = 'random_forest' if results['random_forest']['accuracy'] >= results[
        'logistic_regression']['accuracy'] else 'logistic_regression'
    best_result = results[best_model]
    best_name = 'Random Forest' if best_model == 'random_forest' else 'Logistic Regression'

    print(f"Best performing model: {best_name}")
    print(f"Test Accuracy: {best_result['accuracy']:.4f}")
    print(f"Test F1-Score: {best_result['f1_score']:.4f}")

    return results, df_features


if __name__ == "__main__":
    results, df = main()
