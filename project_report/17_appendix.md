# APPENDIX

---

## Appendix A: Source Code Snippets

### A.1 Complete Feature Engineering Function

```python
def create_features_for_enzyme_prediction(df):
    """
    Create comprehensive feature set for enzyme classification.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw protein dataframe with columns: Entry, Length, EC number, 
        Keywords, Gene Ontology
    
    Returns:
    --------
    pandas.DataFrame
        Enhanced dataframe with engineered features
    """
    import pandas as pd
    import numpy as np
    
    features_df = df.copy()
    
    # Target variable
    features_df['has_ec_number'] = features_df['EC number'].notna()
    
    # Numerical features - Length transformations
    features_df['log_length'] = np.log(features_df['Length'])
    features_df['length_squared'] = features_df['Length'] ** 2
    features_df['length_percentile'] = features_df['Length'].rank(pct=True)
    
    # Text-based count features
    features_df['keyword_count'] = features_df['Keywords'].fillna('').str.count(';') + 1
    features_df['keyword_count'] = features_df['keyword_count'].where(
        features_df['Keywords'].notna(), 0)
    
    features_df['go_term_count'] = features_df['Gene Ontology'].fillna('').str.count('\[GO:')
    
    # Boolean presence features
    features_df['has_keywords'] = features_df['Keywords'].notna()
    features_df['has_go_terms'] = features_df['Gene Ontology'].notna()
    
    # Flag features
    features_df['flag_very_short'] = (features_df['Length'] < 100)
    features_df['flag_very_long'] = (features_df['Length'] > 1000)
    features_df['flag_no_annotation'] = (~features_df['has_keywords']) & \
                                         (~features_df['has_go_terms'])
    
    # Keyword category features
    def has_keyword_type(keywords, keyword_list):
        if pd.isna(keywords):
            return False
        keywords_lower = keywords.lower()
        return any(kw in keywords_lower for kw in keyword_list)
    
    structural_keywords = ['3d-structure', 'domain', 'repeat', 'membrane', 'signal']
    metabolic_keywords = ['metabolism', 'biosynthesis', 'degradation', 'pathway']
    cellular_keywords = ['cell', 'cytoplasm', 'nucleus', 'secreted', 'mitochondria']
    functional_keywords = ['binding', 'transport', 'signaling', 'regulation']
    
    features_df['has_structural_keywords'] = features_df['Keywords'].apply(
        lambda x: has_keyword_type(x, structural_keywords))
    features_df['has_metabolic_keywords'] = features_df['Keywords'].apply(
        lambda x: has_keyword_type(x, metabolic_keywords))
    features_df['has_cellular_keywords'] = features_df['Keywords'].apply(
        lambda x: has_keyword_type(x, cellular_keywords))
    features_df['has_functional_keywords'] = features_df['Keywords'].apply(
        lambda x: has_keyword_type(x, functional_keywords))
    
    # Derived annotation score
    features_df['basic_annotation_score'] = (
        features_df['has_keywords'].astype(int) + 
        features_df['has_go_terms'].astype(int) + 
        features_df['has_ec_number'].astype(int)
    ) / 3
    
    # Length categories
    def categorize_length(length):
        if length < 100:
            return 'Very_Short'
        elif length < 300:
            return 'Short'
        elif length < 500:
            return 'Medium'
        elif length < 1000:
            return 'Long'
        else:
            return 'Very_Long'
    
    features_df['length_category'] = features_df['Length'].apply(categorize_length)
    
    # Convert boolean to int for ML models
    bool_cols = ['has_keywords', 'has_go_terms', 'has_ec_number',
                 'flag_very_short', 'flag_very_long', 'flag_no_annotation',
                 'has_structural_keywords', 'has_metabolic_keywords',
                 'has_cellular_keywords', 'has_functional_keywords']
    
    for col in bool_cols:
        features_df[col] = features_df[col].astype(int)
    
    return features_df
```

### A.2 Model Training and Evaluation Pipeline

```python
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name='Random Forest'):
    """
    Train and evaluate a classification model with comprehensive metrics.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score, accuracy_score)
    import numpy as np
    
    # Initialize model
    if model_name == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1
        )
    else:  # Logistic Regression
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
    
    # Train model
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Test Performance")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Non-Enzyme', 'Enzyme']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance (Random Forest only)
    if model_name == 'Random Forest':
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Important Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    return model, {'accuracy': accuracy, 'roc_auc': roc_auc}
```

### A.3 Visualization Functions

```python
def plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix'):
    """Create a heatmap visualization of confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Enzyme', 'Enzyme'],
                yticklabels=['Non-Enzyme', 'Enzyme'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{title.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, title='ROC Curve'):
    """Plot ROC curve with AUC score."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{title.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10):
    """Plot top N feature importances."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('outputs/plots/feature_importance.png', dpi=300)
    plt.show()
```

---

## Appendix B: Additional Visualizations

### B.1 Data Distribution Plots

The following visualizations were generated during exploratory data analysis:

1. **Protein Length Distribution**
   - Histogram showing right-skewed distribution
   - Mean: 542 amino acids
   - Median: 478 amino acids
   - Range: 50-1,488 amino acids

2. **Data Completeness Bar Chart**
   - Entry: 100% complete
   - Length: 100% complete
   - Keywords: 100% complete
   - Gene Ontology: 92.2% complete
   - EC number: 57.2% complete

3. **Length Categories Pie Chart**
   - Very Short (<100 AA): 8.8%
   - Short (100-300 AA): 23.8%
   - Medium (300-500 AA): 28.0%
   - Long (500-1000 AA): 32.2%
   - Very Long (>1000 AA): 7.2%

### B.2 Model Performance Visualizations

**ROC Curves**:
- Logistic Regression: AUC = 0.8495
- Random Forest: AUC = 0.9599

**Confusion Matrices**: 
- Visual heatmaps showing prediction accuracy
- Random Forest: 89.31% accuracy
- Logistic Regression: 76.03% accuracy

**Feature Importance Plot**:
- Top 10 features ranked by Gini importance
- keyword_count (0.1532) and go_term_count (0.1488) dominate

---

## Appendix C: Dataset Sample

### Sample Protein Entries

```
Entry: A0A009IHW8
Length: 269 amino acids
EC number: 3.2.2.-; 3.2.2.6
Keywords: 3D-structure; Coiled coil; Hydrolase; NAD
GO Terms: NAD+ nucleosidase activity [GO:0003953]; ...

Entry: A0A023I7E1
Length: 796 amino acids
EC number: 3.2.1.39
Keywords: 3D-structure; Carbohydrate metabolism; Cell wall; Glycosidase
GO Terms: endo-1,3(4)-beta-glucanase activity [GO:0052861]; ...

Entry: Q9Y6K9
Length: 478 amino acids
EC number: (none)
Keywords: Alternative splicing; Cytoplasm; Phosphoprotein; Reference proteome
GO Terms: protein binding [GO:0005515]; cytoplasm [GO:0005737]; ...
```

---

## Appendix D: Performance Summary Table

| Model | Task | Accuracy | ROC-AUC | Training Time | Notes |
|-------|------|----------|---------|---------------|-------|
| Logistic Regression | Enzyme | 76.03% | 0.8495 | 0.2s | Fast baseline |
| Random Forest | Enzyme | 89.31% | 0.9599 | 2.1s | Best performer |
| Logistic Regression | Length | 97.75% | N/A | 0.3s | Multi-class |
| Random Forest | Length | 99.50% | N/A | 2.5s | Near perfect |

---

## Appendix E: Glossary

**EC Number**: Enzyme Commission number, a hierarchical classification system for enzymes

**Gene Ontology (GO)**: A standardized vocabulary for describing gene and protein functions

**ROC-AUC**: Receiver Operating Characteristic - Area Under Curve, measures classifier performance

**Random Forest**: An ensemble learning method using multiple decision trees

**Feature Engineering**: The process of creating new features from raw data

**Cross-Validation**: A technique to assess model generalization by splitting data into folds

**Gini Impurity**: A measure of node purity in decision trees

**Data Leakage**: When information from outside the training dataset influences the model

---
