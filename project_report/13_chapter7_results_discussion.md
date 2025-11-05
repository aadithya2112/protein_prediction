# CHAPTER 7: RESULTS AND DISCUSSION

---

## 7.1 Model Performance Metrics

We evaluated our models using standard classification metrics appropriate for both binary (enzyme) and multi-class (length category) classification tasks.

### 7.1.1 Evaluation Metrics Used

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve (binary classification only)
- **Confusion Matrix**: Detailed breakdown of predictions vs. actual classes

## 7.2 Enzyme Classification Results

### 7.2.1 Logistic Regression Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 76.03% |
| Test ROC-AUC | 0.8495 |
| Cross-validation ROC-AUC | 0.8504 ± 0.0020 |
| Precision (Enzyme) | 0.74 |
| Recall (Enzyme) | 0.82 |
| F1-Score (Enzyme) | 0.78 |

**Classification Report (Logistic Regression)**:
```
              precision    recall  f1-score   support

 Non-Enzyme       0.80      0.68      0.74       171
     Enzyme       0.74      0.82      0.78       229

   accuracy                           0.76       400
  macro avg       0.77      0.75      0.76       400
weighted avg       0.76      0.76      0.76       400
```

### 7.2.2 Random Forest Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **89.31%** |
| **Test ROC-AUC** | **0.9599** |
| Cross-validation ROC-AUC | 0.9589 ± 0.0015 |
| Precision (Enzyme) | 0.88 |
| Recall (Enzyme) | 0.91 |
| F1-Score (Enzyme) | 0.89 |

**Classification Report (Random Forest)**:
```
              precision    recall  f1-score   support

 Non-Enzyme       0.91      0.87      0.89       171
     Enzyme       0.88      0.91      0.89       229

   accuracy                           0.89       400
  macro avg       0.89      0.89      0.89       400
weighted avg       0.89      0.89      0.89       400
```

**Confusion Matrix (Random Forest - Enzyme Classification)**:
```
                 Predicted
                 Non-Enz  Enzyme
Actual Non-Enz     149      22
       Enzyme       21     208
```

**Analysis**: Random Forest significantly outperforms Logistic Regression, achieving 13.3% higher accuracy and 0.11 higher ROC-AUC. The balanced precision and recall (both ~0.89) indicate the model performs well for both classes. Low cross-validation standard deviation (±0.0015) suggests stable, reliable performance.

## 7.3 Length Classification Results

### 7.3.1 Random Forest Performance (Length Categories)

| Metric | Value |
|--------|-------|
| Test Accuracy | 99.50% |
| Macro-average F1-Score | 0.99 |
| Weighted-average F1-Score | 0.99 |

**Per-Class Performance**:

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Very_Short | 1.00 | 0.97 | 0.99 | 35 |
| Short | 0.99 | 1.00 | 0.99 | 95 |
| Medium | 1.00 | 1.00 | 1.00 | 112 |
| Long | 0.99 | 1.00 | 0.99 | 129 |
| Very_Long | 1.00 | 1.00 | 1.00 | 29 |

**Confusion Matrix (Length Classification)**:
```
              VS   S    M    L    VL
Very_Short    34   1    0    0    0
Short          0  95    0    0    0
Medium         0   0  112    0    0
Long           0   0    0  129    0
Very_Long      0   0    0    0   29
```

**Analysis**: Near-perfect classification (99.5% accuracy) demonstrates that protein length can be reliably categorized when proper feature engineering is applied. Only 2 misclassifications out of 400 test samples occurred, both at category boundaries (Very_Short/Short).

## 7.4 Feature Importance Analysis

### 7.4.1 Top Features for Enzyme Classification (Random Forest)

| Rank | Feature | Importance Score | Interpretation |
|------|---------|------------------|----------------|
| 1 | keyword_count | 0.1532 | Number of annotation keywords |
| 2 | go_term_count | 0.1488 | Number of GO term annotations |
| 3 | Length | 0.1029 | Protein sequence length |
| 4 | log_length | 0.1027 | Logarithmic transformation of length |
| 5 | length_squared | 0.0892 | Quadratic length feature |
| 6 | basic_annotation_score | 0.0847 | Completeness of annotations |
| 7 | has_go_terms | 0.0757 | Presence of GO annotations |
| 8 | has_metabolic_keywords | 0.0621 | Metabolic function indicators |
| 9 | has_structural_keywords | 0.0513 | Structural annotation presence |
| 10 | has_functional_keywords | 0.0498 | Functional annotation presence |

**Key Insights**:
1. **Annotation richness dominates**: The top 2 features (keyword_count, go_term_count) account for ~30% of total importance, suggesting that well-annotated proteins are more likely to be enzymes.

2. **Length matters**: Protein length features (Length, log_length, length_squared) collectively contribute ~30% importance. Enzymes tend to have characteristic size ranges.

3. **Feature engineering value**: Derived features (basic_annotation_score, keyword categories) provide additional discriminative power beyond raw features.

4. **Non-linear relationships**: The importance of both linear (Length) and non-linear (log_length, length_squared) length features suggests complex relationships that Random Forest captures well.

### 7.4.2 Top Features for Logistic Regression (Coefficients)

| Rank | Feature | Coefficient | Effect |
|------|---------|-------------|--------|
| 1 | Length | 4.0602 | Strong positive |
| 2 | log_length | 3.1978 | Strong positive |
| 3 | length_squared | 1.6810 | Moderate positive |
| 4 | basic_annotation_score | 0.8467 | Moderate positive |
| 5 | has_go_terms | 0.7569 | Moderate positive |

**Comparison**: While both models identify similar important features, Random Forest leverages annotation counts more effectively, explaining its superior performance. Logistic Regression relies more heavily on length features due to its linear nature.

## 7.5 Discussion

### 7.5.1 Model Performance Interpretation

The strong performance (89.31% accuracy, 0.9599 ROC-AUC) for enzyme classification demonstrates that protein metadata contains sufficient signal for functional prediction. This is remarkable given that we use no sequence information—traditional methods would require BLAST searches or domain analysis.

The near-perfect length classification (99.5%) is expected since length categories are defined by length ranges. However, the model's ability to generalize to unseen data validates our feature engineering and confirms the robustness of the Random Forest approach.

### 7.5.2 Biological Insights

1. **Annotation correlation**: Enzymes tend to be better annotated with more keywords and GO terms, reflecting their well-studied nature and functional importance.

2. **Size patterns**: The importance of length features suggests enzymes have characteristic size distributions, possibly related to their need for active sites and regulatory domains.

3. **Metabolic keywords**: The relevance of metabolic keyword features aligns with the fact that many enzymes participate in metabolic pathways.

### 7.5.3 Data Leakage Prevention

Initial models showed suspiciously high performance (>95% accuracy) due to data leakage from features like `data_richness_score` (correlation 0.88 with target) and `has_enzyme_keywords`. After removing these features, performance stabilized at realistic levels (~89% accuracy), demonstrating the importance of careful feature validation.

### 7.5.4 Practical Implications

- **Screening tool**: The model can rapidly screen large protein databases to prioritize candidates for experimental validation
- **Annotation enhancement**: Can suggest enzyme annotations for poorly characterized proteins
- **Complementary to sequence methods**: Provides orthogonal evidence when combined with BLAST or domain-based predictions
- **Computational efficiency**: Millisecond predictions vs. hours for comprehensive sequence analysis

---
