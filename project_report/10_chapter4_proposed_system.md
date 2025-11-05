# CHAPTER 4: PROPOSED SYSTEM

---

## 4.1 System Overview

Our proposed system uses machine learning to classify proteins based on metadata features rather than sequence information. The system addresses two classification problems:

1. **Enzyme Classification (Binary)**: Predict whether a protein is an enzyme (has EC number)
2. **Length Categorization (Multi-class)**: Classify proteins into size categories (Very Short, Short, Medium, Long, Very Long)

**Key Innovation**: Unlike sequence-dependent methods, our approach works with protein metadata (length, keywords, GO terms), making it applicable when sequences are unavailable and computationally efficient for large-scale screening.

## 4.2 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT DATA LAYER                         │
│         UniProt Database → proteins.tsv (2,000 samples)      │
│    Features: Entry, Length, EC number, Keywords, GO terms    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA PREPROCESSING LAYER                    │
│  • Missing value handling     • Text parsing                 │
│  • Feature extraction         • Data type conversion         │
│  • Outlier detection          • Data validation              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING LAYER                    │
│  Numerical: length, log_length, keyword_count, go_term_count│
│  Boolean: has_keywords, has_go_terms, flag_very_short        │
│  Categorical: length_category, keyword_types                 │
│  Derived: annotation_score, length_percentile                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   MACHINE LEARNING LAYER                     │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │ Logistic         │         │ Random Forest    │          │
│  │ Regression       │         │ Classifier       │          │
│  │ (Baseline)       │         │ (Primary Model)  │          │
│  └──────────────────┘         └──────────────────┘          │
│         Train/Test Split (80/20) + Cross-Validation         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION LAYER                           │
│  Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC    │
│  Visualizations: Confusion Matrix, ROC Curve, Feature Imp.   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                             │
│  • Classification predictions    • Confidence scores         │
│  • Performance reports           • Feature importance        │
│  • Model artifacts (.pkl files)  • Visualizations           │
└─────────────────────────────────────────────────────────────┘
```

## 4.3 Workflow

**Step 1: Data Collection**
- Load protein data from UniProt-derived TSV file
- Verify data completeness and quality

**Step 2: Exploratory Data Analysis**
- Statistical analysis of protein lengths
- Distribution analysis of annotations
- Correlation analysis between features

**Step 3: Feature Engineering**
- Extract numerical features (counts, lengths, percentiles)
- Create boolean indicators (has_keywords, has_go_terms)
- Generate derived features (annotation scores, log transformations)
- Encode categorical variables

**Step 4: Data Leakage Prevention**
- Identify and remove features with perfect correlation to target
- Exclude enzyme-specific keywords for enzyme classification
- Validate feature independence

**Step 5: Model Training**
- Split data into training (80%) and test (20%) sets
- Train Logistic Regression (baseline)
- Train Random Forest (primary model)
- Apply cross-validation for robustness

**Step 6: Model Evaluation**
- Calculate performance metrics
- Generate confusion matrices
- Plot ROC curves
- Analyze feature importance

**Step 7: Results Generation**
- Create performance reports
- Visualize results
- Save trained models

## 4.4 Algorithms Used

### 4.4.1 Random Forest Classifier (Primary)

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of classes (classification) from individual trees.

**Advantages**:
- Handles non-linear relationships effectively
- Robust to outliers and overfitting
- Provides feature importance scores
- No feature scaling required
- Handles mixed data types well

**Configuration**:
- Number of trees: 100
- Max depth: Auto
- Min samples split: 2
- Criterion: Gini impurity

### 4.4.2 Logistic Regression (Baseline)

A linear model for binary classification that estimates probabilities using a logistic function.

**Advantages**:
- Fast training and prediction
- Interpretable coefficients
- Provides probability scores
- Works well with linearly separable data

**Configuration**:
- Solver: lbfgs
- Max iterations: 1000
- Regularization: L2 (default)

## 4.5 Expected Benefits

1. **Speed**: Classification in milliseconds vs. hours for BLAST searches
2. **Scalability**: Can process millions of proteins efficiently
3. **Metadata-Only**: Works without sequence information
4. **Probabilistic**: Provides confidence scores, not just binary predictions
5. **Interpretable**: Feature importance reveals biological insights
6. **Automated**: No manual curation required
7. **Complementary**: Can be combined with sequence-based methods for improved accuracy
8. **Cost-Effective**: Minimal computational resources compared to deep learning approaches

---
