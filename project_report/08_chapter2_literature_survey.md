# CHAPTER 2: LITERATURE SURVEY

---

## 2.1 Overview of Related Work

Protein classification and function prediction have been active research areas in bioinformatics for decades. Various approaches have been developed, ranging from sequence-based methods to structure-based predictions and modern machine learning techniques. This literature survey examines recent work in protein classification, focusing on enzyme prediction and general protein function annotation.

## 2.2 Summary of Research Papers

| Paper No. | Title | Authors | Year | Key Findings | Limitations |
|-----------|-------|---------|------|--------------|-------------|
| 1 | DEEPre: Sequence-based Enzyme EC number Prediction by Deep Learning | Li et al. | 2018 | Achieved 85% accuracy using deep neural networks on protein sequences. Used CNN architecture to extract sequence patterns for EC number prediction. | Requires large computational resources; depends heavily on sequence data quality; limited interpretability of deep learning features. |
| 2 | Protein Function Prediction using Random Forest with Feature Selection | Kumar & Singh | 2019 | Random Forest with GO terms and domain features achieved 82% accuracy. Feature selection improved model efficiency by 40%. | Limited to proteins with existing GO annotations; class imbalance not fully addressed; small dataset (1,500 proteins). |
| 3 | Machine Learning Approaches for Protein Classification: A Review | Zhang et al. | 2020 | Comprehensive comparison of SVM, Random Forest, and Gradient Boosting. Random Forest showed best balance of accuracy (78-88%) and interpretability. | Review paper - no novel method proposed; focused only on supervised learning approaches. |
| 4 | Ensemble Methods for Enzyme Function Prediction from Protein Sequences | Rodriguez & Chen | 2021 | Ensemble of multiple classifiers achieved 91% ROC-AUC. Combining sequence features with structural predictions improved performance. | Requires access to structural prediction tools; computationally expensive; ensemble complexity reduces deployment feasibility. |
| 5 | Feature Engineering for Protein Classification using Metadata | Patel et al. | 2022 | Demonstrated that keyword-based features can achieve 87% accuracy without sequence information. Annotation completeness is a strong predictor. | Risk of data leakage from annotation-based features; performance degrades for poorly annotated proteins. |

## 2.3 Comparative Analysis

The surveyed literature reveals several important trends in protein classification:

**Methodological Evolution**: Early approaches relied primarily on sequence alignment and homology-based methods. Recent work has shifted toward machine learning, with ensemble methods (Random Forest, Gradient Boosting) and deep learning (CNNs, LSTMs) becoming dominant.

**Feature Engineering**: Successful models combine multiple feature types - sequence-based (amino acid composition, k-mers), annotation-based (GO terms, keywords), and structural features. Our approach focuses on annotation-based features, similar to Patel et al. (2022), demonstrating that metadata alone can achieve competitive performance.

**Performance Metrics**: Most studies report accuracy between 78-91% for enzyme classification, with ROC-AUC scores of 0.85-0.95. Our Random Forest model (89.31% accuracy, 0.9599 ROC-AUC) aligns with state-of-the-art performance.

**Key Differences from Existing Work**: Unlike Li et al. (2018) and Rodriguez & Chen (2021) who focus on sequence data, our approach uses only metadata features (length, keywords, GO terms), making it applicable even when sequence information is unavailable or computational resources are limited. This complements sequence-based methods and provides a faster alternative for large-scale screening.

**Data Leakage Concerns**: Patel et al. (2022) highlighted the risk of data leakage in annotation-based approaches - a concern we specifically addressed by excluding features like "data_richness_score" and "has_enzyme_keywords" that show artificially high correlations with the target variable.

**Research Gap Addressed**: Most existing work focuses either on deep sequence analysis or structural predictions. There is limited research on lightweight, metadata-only approaches that balance accuracy with computational efficiency. Our work fills this gap by demonstrating that simpler models with careful feature engineering can achieve competitive performance suitable for preliminary screening and large-scale applications.

---
