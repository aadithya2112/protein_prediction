#!/usr/bin/env python3
"""
Enzyme Classification - Data Leakage Investigation
==================================================

This script investigates why the model achieved perfect performance
and identifies potential data leakage issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def investigate_features(df):
    """Investigate potential data leakage in features."""
    print("=== INVESTIGATING DATA LEAKAGE ===")

    # Check correlation between features and target
    print("\n1. Feature correlations with target (has_ec_number):")

    # Convert boolean columns to numeric for correlation and select only numeric columns
    df_numeric = df.copy()
    bool_cols = df.select_dtypes(include=[bool]).columns
    for col in bool_cols:
        df_numeric[col] = df_numeric[col].astype(int)

    # Select only numeric columns for correlation
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
    df_for_corr = df_numeric[numeric_cols]

    # Calculate correlations with target
    target_corr = df_for_corr.corr(
    )['has_ec_number'].abs().sort_values(ascending=False)
    print(target_corr.head(10))

    # Check if any features are perfect predictors
    print("\n2. Perfect predictors (correlation = 1.0):")
    perfect_predictors = target_corr[target_corr == 1.0]
    print(perfect_predictors)

    # Investigate has_enzyme_keywords feature
    if 'has_enzyme_keywords' in df.columns:
        print("\n3. has_enzyme_keywords vs has_ec_number crosstab:")
        crosstab = pd.crosstab(
            df['has_enzyme_keywords'], df['has_ec_number'], margins=True)
        print(crosstab)

        # Check if this is causing the perfect prediction
        print("\n4. Percentage of enzymes with enzyme keywords:")
        enzyme_proteins = df[df['has_ec_number'] == True]
        non_enzyme_proteins = df[df['has_ec_number'] == False]

        print(
            f"Enzymes with enzyme keywords: {enzyme_proteins['has_enzyme_keywords'].sum()} / {len(enzyme_proteins)} ({enzyme_proteins['has_enzyme_keywords'].mean()*100:.1f}%)")
        print(
            f"Non-enzymes with enzyme keywords: {non_enzyme_proteins['has_enzyme_keywords'].sum()} / {len(non_enzyme_proteins)} ({non_enzyme_proteins['has_enzyme_keywords'].mean()*100:.1f}%)")

    return target_corr


def create_realistic_features(df):
    """Create features that don't leak target information."""
    print("\n=== CREATING REALISTIC FEATURES ===")

    features_df = df.copy()

    # Convert boolean columns to integers (excluding target)
    bool_cols = ['has_keywords', 'has_go_terms',
                 'flag_very_short', 'flag_very_long', 'flag_no_annotation']
    for col in bool_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int)

    # Basic numerical features
    features_df['log_length'] = np.log1p(features_df['Length'])
    features_df['length_squared'] = features_df['Length'] ** 2
    features_df['length_percentile'] = features_df['Length'].rank(pct=True)

    # Text-based features (but NOT enzyme-specific keywords!)
    if 'Keywords' in features_df.columns:
        # General keyword count
        features_df['keyword_count'] = features_df['Keywords'].fillna(
            '').str.count(';') + 1
        features_df['keyword_count'] = features_df['keyword_count'].where(
            features_df['Keywords'].notna(), 0)

        # More general keyword categories (avoid enzyme-specific terms)
        structural_keywords = ['3d-structure',
                               'structure', 'domain', 'repeat', 'membrane']
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

    # Gene Ontology based features (general categories)
    if 'Gene_Ontology' in features_df.columns:
        features_df['go_term_count'] = features_df['Gene_Ontology'].fillna(
            '').str.count(';') + 1
        features_df['go_term_count'] = features_df['go_term_count'].where(
            features_df['Gene_Ontology'].notna(), 0)

    # Annotation completeness
    annotation_cols = ['has_keywords', 'has_go_terms']
    features_df['annotation_completeness'] = features_df[annotation_cols].sum(
        axis=1)

    # Length-based categories (encoded)
    length_category_map = {'Very_Short': 0, 'Short': 1,
                           'Medium': 2, 'Long': 3, 'Very_Long': 4}
    features_df['length_category_encoded'] = features_df['length_category'].map(
        length_category_map)

    print(f"Created realistic features. Shape: {features_df.shape}")
    return features_df


def train_realistic_models(df):
    """Train models with realistic features that don't leak target information."""
    print("\n=== TRAINING MODELS WITH REALISTIC FEATURES ===")

    # Define feature columns (exclude leaky features)
    exclude_cols = ['Entry', 'EC_number', 'Keywords', 'Gene_Ontology', 'length_category',
                    'has_ec_number', 'has_enzyme_keywords']  # Exclude the leaky feature!

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Using {len(feature_cols)} realistic features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['has_ec_number'].astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    print("\n--- Logistic Regression Results ---")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    lr_accuracy = (y_pred_lr == y_test).mean()
    lr_roc_auc = roc_auc_score(y_test, y_pred_proba_lr)

    print(f"Accuracy: {lr_accuracy:.4f}")
    print(f"ROC-AUC: {lr_roc_auc:.4f}")
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

    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

    rf_accuracy = (y_pred_rf == y_test).mean()
    rf_roc_auc = roc_auc_score(y_test, y_pred_proba_rf)

    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"ROC-AUC: {rf_roc_auc:.4f}")
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
            'feature_importance': feature_importance_lr
        },
        'random_forest': {
            'model': rf,
            'accuracy': rf_accuracy,
            'roc_auc': rf_roc_auc,
            'feature_importance': feature_importance_rf
        }
    }

    return results, X_test, y_test, feature_cols


def main():
    """Main function for investigating and fixing data leakage."""
    print("ENZYME CLASSIFICATION - DATA LEAKAGE INVESTIGATION")
    print("=" * 60)

    # Load data
    df = pd.read_csv('data/proteins_cleaned.tsv', sep='\t')

    # Add the problematic feature that caused perfect performance
    if 'Keywords' in df.columns:
        enzyme_keywords = ['enzyme', 'kinase', 'phosphatase', 'dehydrogenase',
                           'transferase', 'hydrolase', 'oxidase', 'reductase']
        df['has_enzyme_keywords'] = df['Keywords'].fillna('').str.lower().str.contains(
            '|'.join(enzyme_keywords), regex=True).astype(int)

    # Investigate data leakage
    correlations = investigate_features(df)

    # Create realistic features
    df_realistic = create_realistic_features(df)

    # Train models with realistic features
    results, X_test, y_test, feature_cols = train_realistic_models(
        df_realistic)

    return results, df_realistic


if __name__ == "__main__":
    results, df = main()
