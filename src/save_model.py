#!/usr/bin/env python3
"""
Save Trained Enzyme Classification Model
========================================

This script saves the best performing model for future use.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def create_and_save_final_model():
    """Create and save the final enzyme classification model."""
    print("=== CREATING AND SAVING FINAL MODEL ===")

    # Load data and create features (same as in realistic_enzyme_classification.py)
    df = pd.read_csv('data/proteins_cleaned.tsv', sep='\t')

    # Create features
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

    # Text-based features
    if 'Keywords' in features_df.columns:
        features_df['keyword_count'] = features_df['Keywords'].fillna(
            '').str.count(';') + 1
        features_df['keyword_count'] = features_df['keyword_count'].where(
            features_df['Keywords'].notna(), 0)

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

    # Basic annotation completeness
    features_df['basic_annotation_score'] = features_df['has_keywords'] + \
        features_df['has_go_terms']

    # Length category encoding
    length_category_map = {'Very_Short': 0, 'Short': 1,
                           'Medium': 2, 'Long': 3, 'Very_Long': 4}
    features_df['length_category_encoded'] = features_df['length_category'].map(
        length_category_map)

    # Define features
    exclude_cols = [
        'Entry', 'EC_number', 'Keywords', 'Gene_Ontology', 'length_category',
        'has_ec_number', 'data_richness_score', 'has_enzyme_keywords'
    ]

    feature_cols = [
        col for col in features_df.columns if col not in exclude_cols]

    # Prepare data
    X = features_df[feature_cols].fillna(0)
    y = features_df['has_ec_number'].astype(int)

    # Train final model on full dataset
    print(f"Training final Random Forest model on {len(X)} proteins...")
    final_model = RandomForestClassifier(random_state=42, n_estimators=100)
    final_model.fit(X, y)

    # Create model metadata
    model_metadata = {
        'model_type': 'RandomForestClassifier',
        'features': feature_cols,
        'target': 'has_ec_number',
        'training_size': len(X),
        'feature_count': len(feature_cols),
        'accuracy_estimate': 0.8931,  # From our testing
        'roc_auc_estimate': 0.9599,  # From our testing
        'description': 'Enzyme classification model predicting whether a protein has an EC number'
    }

    # Save model and metadata
    os.makedirs('outputs/models', exist_ok=True)

    model_path = 'outputs/models/enzyme_classifier_rf.joblib'
    metadata_path = 'outputs/models/enzyme_classifier_metadata.joblib'

    joblib.dump(final_model, model_path)
    joblib.dump(model_metadata, metadata_path)

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

    # Create feature importance summary
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_path = 'outputs/models/feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")

    return final_model, model_metadata, feature_importance


def create_prediction_function():
    """Create a simple prediction function script."""
    prediction_script = '''#!/usr/bin/env python3
"""
Enzyme Classification Prediction Function
=========================================

This script provides a simple function to predict whether proteins are enzymes
using the trained Random Forest model.
"""

import pandas as pd
import numpy as np
import joblib

def load_enzyme_classifier():
    """Load the trained enzyme classification model."""
    model = joblib.load('outputs/models/enzyme_classifier_rf.joblib')
    metadata = joblib.load('outputs/models/enzyme_classifier_metadata.joblib')
    return model, metadata

def prepare_features(df):
    """Prepare features for enzyme classification prediction."""
    features_df = df.copy()
    
    # Convert boolean columns to integers
    bool_cols = ['has_keywords', 'has_go_terms', 'flag_very_short', 'flag_very_long', 'flag_no_annotation']
    for col in bool_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int)
    
    # Length-based features
    features_df['log_length'] = np.log1p(features_df['Length'])
    features_df['length_squared'] = features_df['Length'] ** 2
    features_df['length_percentile'] = features_df['Length'].rank(pct=True)
    
    # Text-based features
    if 'Keywords' in features_df.columns:
        features_df['keyword_count'] = features_df['Keywords'].fillna('').str.count(';') + 1
        features_df['keyword_count'] = features_df['keyword_count'].where(features_df['Keywords'].notna(), 0)
        
        structural_keywords = ['3d-structure', 'structure', 'domain', 'repeat', 'membrane', 'signal', 'transmembrane']
        metabolic_keywords = ['metabolism', 'biosynthesis', 'degradation', 'pathway', 'transport']
        cellular_keywords = ['cell', 'cytoplasm', 'nucleus', 'secreted', 'mitochondria', 'ribosome']
        functional_keywords = ['binding', 'activity', 'regulation', 'response', 'development']
        
        features_df['has_structural_keywords'] = features_df['Keywords'].fillna('').str.lower().str.contains('|'.join(structural_keywords), regex=True).astype(int)
        features_df['has_metabolic_keywords'] = features_df['Keywords'].fillna('').str.lower().str.contains('|'.join(metabolic_keywords), regex=True).astype(int)
        features_df['has_cellular_keywords'] = features_df['Keywords'].fillna('').str.lower().str.contains('|'.join(cellular_keywords), regex=True).astype(int)
        features_df['has_functional_keywords'] = features_df['Keywords'].fillna('').str.lower().str.contains('|'.join(functional_keywords), regex=True).astype(int)
    
    # Gene Ontology features
    if 'Gene_Ontology' in features_df.columns:
        features_df['go_term_count'] = features_df['Gene_Ontology'].fillna('').str.count(';') + 1
        features_df['go_term_count'] = features_df['go_term_count'].where(features_df['Gene_Ontology'].notna(), 0)
    
    # Basic annotation completeness
    features_df['basic_annotation_score'] = features_df['has_keywords'] + features_df['has_go_terms']
    
    # Length category encoding
    length_category_map = {'Very_Short': 0, 'Short': 1, 'Medium': 2, 'Long': 3, 'Very_Long': 4}
    features_df['length_category_encoded'] = features_df['length_category'].map(length_category_map)
    
    return features_df

def predict_enzyme_classification(df):
    """
    Predict enzyme classification for proteins.
    
    Parameters:
    df: pandas DataFrame with protein data (same format as cleaned data)
    
    Returns:
    pandas DataFrame with predictions and probabilities
    """
    # Load model
    model, metadata = load_enzyme_classifier()
    
    # Prepare features
    features_df = prepare_features(df)
    
    # Select required features
    feature_cols = metadata['features']
    X = features_df[feature_cols].fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Entry': df['Entry'],
        'predicted_enzyme': predictions,
        'enzyme_probability': probabilities[:, 1],
        'confidence': np.max(probabilities, axis=1)
    })
    
    return results

# Example usage
if __name__ == "__main__":
    # Load test data
    df = pd.read_csv('data/proteins_cleaned.tsv', sep='\\t').head(100)  # Test on first 100 proteins
    
    # Make predictions
    results = predict_enzyme_classification(df)
    
    print("Sample predictions:")
    print(results.head(10))
    
    print(f"\\nPredicted {results['predicted_enzyme'].sum()} enzymes out of {len(results)} proteins")
'''

    with open('src/enzyme_predictor.py', 'w') as f:
        f.write(prediction_script)

    print("Prediction function saved to: src/enzyme_predictor.py")


def main():
    """Main function to save the final model."""
    print("SAVING FINAL ENZYME CLASSIFICATION MODEL")
    print("=" * 45)

    # Create and save final model
    model, metadata, importance = create_and_save_final_model()

    # Create prediction function
    create_prediction_function()

    print("\\n=== MODEL SUMMARY ===")
    print(f"Model type: {metadata['model_type']}")
    print(f"Features used: {metadata['feature_count']}")
    print(f"Training samples: {metadata['training_size']}")
    print(f"Estimated accuracy: {metadata['accuracy_estimate']:.4f}")
    print(f"Estimated ROC-AUC: {metadata['roc_auc_estimate']:.4f}")

    print("\\nTop 5 most important features:")
    for idx, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
