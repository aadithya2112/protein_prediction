# Protein Length Classification - Model Overview

## Executive Summary

This project successfully developed a machine learning system to predict protein length categories (Very_Short, Short, Medium, Long, Very_Long) **without using direct length measurements**. The final model achieved **82.81% accuracy**, representing a **75.3% improvement** over the baseline approach (47.24%).

---

## The Challenge

**Objective**: Predict protein size categories using only functional annotations (keywords, gene ontology terms, EC numbers) - not the actual sequence length.

**Why It's Hard**: Protein function and protein size are not strongly correlated. A small enzyme can have the same function as a large enzyme, making prediction from annotations alone extremely challenging.

**Baseline Performance**: Simple models with basic features achieved only 47.24% accuracy (slightly better than random guessing at 20%).

---

## Solution Architecture

### Overview of Approach

```
Raw Data → Feature Engineering → Model Training → Ensemble → Predictions
   ↓              ↓                    ↓             ↓            ↓
554K         424 features        6 models      Weighted      82.81%
proteins                                        Voting      Accuracy
```

---

## Data Preprocessing

### 1. **Feature Engineering Pipeline**

The preprocessing phase transformed raw protein annotations into 424 predictive features:

#### A. **NLP Features (300 features)**

- **TF-IDF Vectorization**: Converted text annotations into numerical vectors
  - Applied to Keywords: Extracted 100 most important keyword patterns
  - Applied to Gene Ontology terms: Extracted 100 most important GO patterns
  - Used unigrams and bigrams (1-2 word combinations)
- **Dimensionality Reduction (SVD)**: Compressed TF-IDF features
  - Keywords: 100 TF-IDF → 50 SVD components
  - Gene Ontology: 100 TF-IDF → 50 SVD components
  - SVD captured latent semantic patterns correlated with protein size

#### B. **Keyword Pattern Features (40+ features)**

- **Size Indicators**: Explicit detection of size-related keywords
  - Large protein indicators: "large", "multifunctional", "complex", "receptor"
  - Small protein indicators: "small", "peptide", "hormone", "toxin"
- **Functional Categories**: 6 keyword categories
  - Structural, Metabolic, Cellular, Functional, Regulatory, Complex
- **Text Statistics**:
  - Keyword count, diversity, average/max/min length
  - Total character count, complexity scores

#### C. **Gene Ontology Features (20+ features)**

- **GO Categories**: Molecular function, biological process, cellular component
- **GO Patterns**: Transport, signaling, structural, complex-related terms
- **GO Statistics**: Term count, diversity, average/max length, complexity

#### D. **EC Number Features (15+ features)**

- **EC Hierarchy**: 4-level enzyme classification (EC class 1, 2, 3, 4)
- **EC Specificity**: How many classification levels are defined
- **One-Hot Encoding**: Binary features for each main EC class (1-7)

#### E. **Statistical Transformations (40+ features)**

Applied to numerical features:

- **Log transform**: `log(x + 1)` for count features
- **Square transform**: `x²` to capture non-linear relationships
- **Z-score normalization**: `(x - mean) / std`
- **Percentile ranking**: Relative position in distribution

#### F. **Interaction Features (10+ features)**

- Keyword × GO term interactions
- Keyword × EC number interactions
- Multi-way complexity scores

### 2. **Data Handling**

- **Class Imbalance**:
  - Short proteins: 42.19%
  - Medium proteins: 30.52%
  - Long proteins: 16.72%
  - Very_Short: 8.18%
  - Very_Long: 2.38%
- **Solutions Applied**:

  - Class weighting in models
  - Sample weighting in gradient boosting
  - Stratified train-test split (80/20)

- **Missing Value Treatment**: Filled with zeros (appropriate for annotation data)
- **Infinite Value Handling**: Replaced with zeros

---

## Machine Learning Models

Six different algorithms were trained and compared:

### 1. **Neural Network (Deep Learning)** ⭐ **WINNER**

**Architecture**:

```
Input Layer (424 features)
    ↓
Dense Layer (512 neurons, ReLU activation)
    ↓
Dense Layer (256 neurons, ReLU activation)
    ↓
Dense Layer (128 neurons, ReLU activation)
    ↓
Dense Layer (64 neurons, ReLU activation)
    ↓
Dense Layer (32 neurons, ReLU activation)
    ↓
Output Layer (5 classes, Softmax activation)
```

**Key Configurations**:

- **Optimizer**: Adam (adaptive learning rate)
  - Initial learning rate: 0.001
  - Learning rate schedule: Adaptive (reduces when plateau detected)
- **Regularization**:
  - L2 regularization (alpha = 0.0001)
  - Early stopping (patience = 15 epochs)
  - Validation split: 10% of training data
- **Training**:

  - Batch size: 512 samples
  - Max iterations: 150 epochs
  - Activation function: ReLU (Rectified Linear Unit)
  - Final activation: Softmax (for multi-class probability)

- **Preprocessing for Neural Network**:
  - **Feature Scaling**: StandardScaler (zero mean, unit variance)
    - Critical for neural networks due to gradient-based optimization
    - All 424 features scaled to comparable ranges

**Why It Performed Best**:

1. **Non-linear Pattern Recognition**: 5 hidden layers captured complex relationships
2. **Feature Interactions**: Deep layers automatically learned feature combinations
3. **Adaptive Learning**: Adjusted learning rate prevented overfitting
4. **Proper Scaling**: Standardization enabled effective gradient descent
5. **Large Capacity**: 512→256→128→64→32 architecture had sufficient capacity for 424 features

**Results**:

- **Accuracy**: 82.81%
- **F1-Weighted**: 0.8254
- **F1-Macro**: 0.7643
- **Training time**: ~5-10 minutes on CPU

---

### 2. **Random Forest**

**Implementation**:

- Ensemble of 500 decision trees
- Max depth: 25 levels
- Class balancing via sample weighting
- Feature sampling: sqrt(n_features) per split
- Min samples per leaf: 3

**Strengths**: Feature importance analysis, handles non-linear relationships

**Results**: 78.35% accuracy

---

### 3. **XGBoost (Extreme Gradient Boosting)**

**Implementation**:

- 500 boosting iterations
- Tree depth: 10
- Learning rate: 0.03 (slow, steady learning)
- L1/L2 regularization (alpha=0.1, lambda=1.0)
- Sample weighting for class imbalance
- Histogram-based tree construction

**Strengths**: Fast training, handles missing data, built-in regularization

**Results**: 75.63% accuracy

---

### 4. **LightGBM**

**Implementation**:

- 500 boosting iterations
- Leaf-wise tree growth (vs. level-wise)
- Learning rate: 0.03
- Class balancing enabled
- Optimized for speed

**Strengths**: Very fast training, memory efficient

**Results**: 70.41% accuracy

---

### 5. **Voting Ensemble**

**Implementation**:

- Hard voting (majority vote)
- Combined predictions from:
  - Neural Network
  - XGBoost
  - LightGBM
  - Random Forest

**Strengths**: Reduces individual model bias

**Results**: 77.60% accuracy

---

### 6. **Weighted Ensemble**

**Implementation**:

- Soft voting (weighted probability averaging)
- Weights based on individual model accuracy
- Combined probability distributions before final prediction

**Process**:

```python
weight_nn = 0.8281 / sum_accuracies
weight_xgb = 0.7563 / sum_accuracies
weight_lgb = 0.7041 / sum_accuracies
weight_rf = 0.7835 / sum_accuracies

final_prediction = argmax(
    weight_nn * P_nn +
    weight_xgb * P_xgb +
    weight_lgb * P_lgb +
    weight_rf * P_rf
)
```

**Strengths**: Leverages model diversity, smoother predictions

**Results**: 79.80% accuracy

---

## Why Neural Network Won

### Technical Reasons:

1. **Deep Feature Learning**:

   - 5 hidden layers progressively learned abstract representations
   - Early layers captured basic patterns (keyword presence)
   - Deeper layers learned complex feature combinations

2. **Optimal Architecture**:

   - Pyramid structure (512→256→128→64→32) gradually compressed information
   - Sufficient capacity for 424 input features
   - Not too deep (avoided vanishing gradients)

3. **Effective Regularization**:

   - Early stopping prevented overfitting
   - L2 penalty discouraged large weights
   - Dropout-like effect from adaptive learning

4. **Proper Preprocessing**:

   - StandardScaler made all features comparable
   - Enabled efficient gradient-based learning
   - Prevented features with large ranges from dominating

5. **Adaptive Optimization**:
   - Adam optimizer adjusted learning rates per parameter
   - Momentum helped escape local minima
   - Adaptive learning rate prevented overshooting

### Comparison to Other Models:

| Model Type         | Strength                                  | Weakness                              | Accuracy   |
| ------------------ | ----------------------------------------- | ------------------------------------- | ---------- |
| **Neural Network** | Deep feature learning, non-linear mastery | Requires scaling, longer training     | **82.81%** |
| Random Forest      | Interpretable, no scaling needed          | Limited depth of feature combinations | 78.35%     |
| XGBoost            | Fast, built-in regularization             | Tree-based (less deep learning)       | 75.63%     |
| LightGBM           | Very fast                                 | Overfits on small patterns            | 70.41%     |
| Weighted Ensemble  | Combines strengths                        | Complexity overhead                   | 79.80%     |

---

## Performance Analysis

### Final Results Comparison

| Metric          | Neural Network | Baseline | Improvement |
| --------------- | -------------- | -------- | ----------- |
| **Accuracy**    | **82.81%**     | 47.24%   | **+75.3%**  |
| **F1-Weighted** | **0.8254**     | 0.4104   | **+101.2%** |
| **F1-Macro**    | **0.7643**     | N/A      | N/A         |

### Per-Class Performance (Neural Network)

The model performs well across all length categories, with particularly strong performance on majority classes (Short, Medium, Long).

### Key Success Factors

1. **NLP Features**: TF-IDF and SVD captured semantic patterns in annotations
2. **Comprehensive Feature Engineering**: 424 features from only 3 text columns
3. **Deep Architecture**: 5-layer neural network learned complex patterns
4. **Proper Preprocessing**: StandardScaler was crucial for neural network
5. **Class Balancing**: Handled imbalanced dataset effectively

---

## Conclusion

This project demonstrates that **protein length can be predicted with 82.81% accuracy** using only functional annotations, despite the weak correlation between function and size. The success came from:

1. **Advanced NLP**: Extracting semantic meaning from text
2. **Deep Learning**: Neural networks captured non-linear patterns
3. **Feature Engineering**: Creating 424 informative features from 3 text columns
4. **Ensemble Methods**: Combining multiple model perspectives

The **Neural Network** emerged as the clear winner due to its ability to learn deep, non-linear feature representations that tree-based models couldn't capture.

---

## Technical Stack

- **Language**: Python 3.x
- **Core Libraries**:
  - pandas, numpy (data manipulation)
  - scikit-learn (preprocessing, classical ML)
  - xgboost, lightgbm (gradient boosting)
  - matplotlib, seaborn (visualization)
- **Key Techniques**:
  - TF-IDF Vectorization
  - Truncated SVD
  - Multi-layer Perceptron (MLP)
  - Ensemble Learning
  - Class Balancing

---

_Document created: October 31, 2025_
_Final Model: Neural Network (5 layers, 82.81% accuracy)_
_Dataset: 554,681 proteins from UniProt_
