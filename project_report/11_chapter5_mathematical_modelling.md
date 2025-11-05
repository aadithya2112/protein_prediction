# CHAPTER 5: MATHEMATICAL MODELLING OF THE ALGORITHM

---

## 5.1 Problem Formulation

### 5.1.1 Input-Output Specification

**Input Space (X)**: Feature vector for each protein
```
x ∈ ℝⁿ where n = number of features
x = [x₁, x₂, ..., xₙ]ᵀ
```

For our implementation:
- x₁ = Length (amino acids)
- x₂ = log(Length)
- x₃ = keyword_count
- x₄ = go_term_count
- x₅...xₙ = boolean and derived features

**Output Space (Y)**: Class labels

For enzyme classification (binary):
```
y ∈ {0, 1} where 0 = non-enzyme, 1 = enzyme
```

For length classification (multi-class):
```
y ∈ {1, 2, 3, 4, 5} representing {Very_Short, Short, Medium, Long, Very_Long}
```

### 5.1.2 Training Dataset

```
D = {(x⁽¹⁾, y⁽¹⁾), (x⁽²⁾, y⁽²⁾), ..., (x⁽ᵐ⁾, y⁽ᵐ⁾)}
```
where m = 2,000 protein samples

### 5.1.3 Objective

Find a function f: X → Y that minimizes prediction error:
```
f*(x) = argmin E[(f(x) - y)²]
        f∈ℱ
```
where ℱ is the hypothesis space (set of possible models)

## 5.2 Random Forest Mathematics

### 5.2.1 Decision Tree Foundation

A single decision tree partitions the feature space recursively:

At each node, find best split:
```
(j*, t*) = argmin [G_left(j,t) + G_right(j,t)]
           j,t

where:
j = feature index
t = threshold value
G = Gini impurity or entropy
```

**Gini Impurity** for node m:
```
G(m) = 1 - Σ(p_k)²
           k=1

where:
K = number of classes
p_k = proportion of class k samples in node m
```

### 5.2.2 Ensemble Aggregation

Random Forest builds B trees using bootstrap samples:

```
For b = 1 to B:
  1. Draw bootstrap sample D_b from D (sampling with replacement)
  2. Grow tree T_b using random feature subset at each split
  3. No pruning applied
```

**Final Prediction** (Classification):
```
ŷ = majority_vote{T₁(x), T₂(x), ..., T_B(x)}

For probabilities:
P(y=k|x) = (1/B) Σ I(T_b(x) = k)
                b=1
```

### 5.2.3 Feature Importance

**Gini Importance** for feature j:
```
Imp(j) = Σ Σ p(t) × ΔG(s_t, j)
         b t∈T_b

where:
p(t) = proportion of samples reaching node t
ΔG(s_t, j) = Gini decrease if splitting on feature j at node t
```

## 5.3 Logistic Regression Mathematics

### 5.3.1 Model Definition

**Logistic Function** (Sigmoid):
```
σ(z) = 1 / (1 + e^(-z))

where z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
```

**Probability Estimation**:
```
P(y=1|x; θ) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
P(y=0|x; θ) = 1 - P(y=1|x; θ)
```

### 5.3.2 Cost Function

**Log-Loss (Binary Cross-Entropy)**:
```
J(θ) = -(1/m) Σ [y⁽ⁱ⁾ log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1-h_θ(x⁽ⁱ⁾))]
              i=1

where:
h_θ(x) = σ(θᵀx) = predicted probability
m = number of training examples
```

**With L2 Regularization**:
```
J(θ) = -(1/m) Σ [y⁽ⁱ⁾ log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1-h_θ(x⁽ⁱ⁾))] + (λ/2m) Σ θⱼ²
              i=1                                                            j=1

where λ = regularization parameter
```

### 5.3.3 Multi-class Extension

For length classification with K=5 classes, we use **One-vs-Rest (OvR)**:

Train K binary classifiers:
```
For k = 1 to K:
  Train classifier to distinguish class k from all others
  θ_k = parameters for classifier k

Prediction:
ŷ = argmax P(y=k|x; θ_k)
    k∈{1,...,K}
```

## 5.4 Optimization

### 5.4.1 Gradient Descent for Logistic Regression

**Update Rule**:
```
θⱼ := θⱼ - α ∂J(θ)/∂θⱼ

where:
∂J(θ)/∂θⱼ = (1/m) Σ (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾ + (λ/m)θⱼ
                  i=1

α = learning rate
```

**L-BFGS Solver**: We use Limited-memory BFGS (quasi-Newton method) which approximates the Hessian matrix for faster convergence.

### 5.4.2 Performance Metrics

**Accuracy**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision & Recall**:
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

**F1-Score** (Harmonic mean):
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**ROC-AUC**: Area under Receiver Operating Characteristic curve
```
AUC = ∫₀¹ TPR(FPR) d(FPR)

where:
TPR = True Positive Rate = TP/(TP+FN)
FPR = False Positive Rate = FP/(FP+TN)
```

### 5.4.3 Cross-Validation

**K-Fold Cross-Validation** (K=5):
```
For fold k = 1 to K:
  1. Use fold k as validation set
  2. Use remaining K-1 folds as training set
  3. Train model and evaluate on fold k
  4. Record performance metric M_k

Average performance: M_avg = (1/K) Σ M_k
Standard deviation: σ_M = sqrt[(1/K) Σ (M_k - M_avg)²]
```

---
