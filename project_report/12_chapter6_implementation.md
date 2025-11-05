# CHAPTER 6: IMPLEMENTATION

---

## 6.1 Tools and Libraries

### 6.1.1 Python Environment

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Core programming language |
| pandas | 2.0.0+ | Data manipulation and analysis |
| numpy | 1.24.0+ | Numerical computations |
| scikit-learn | 1.3.0+ | Machine learning algorithms and metrics |
| matplotlib | 3.6.0+ | Data visualization (static plots) |
| seaborn | 0.12.0+ | Statistical data visualization |
| jupyter | 1.0.0+ | Interactive development environment |

### 6.1.2 Development Environment

- **IDE**: Jupyter Notebook for interactive analysis
- **Version Control**: Git for code management
- **Operating System**: Linux/Windows/macOS compatible
- **Hardware**: Standard CPU sufficient (no GPU required)

## 6.2 Dataset Description

### 6.2.1 Data Source

**Origin**: UniProt Database (Universal Protein Resource)
- Public protein sequence and annotation database
- Comprehensive functional information
- Regularly updated with new protein entries

**Dataset File**: `proteins.tsv` (Tab-separated values)
- Sample size: 2,000 proteins (subset from 554K full dataset)
- File size: ~630 KB
- Character encoding: UTF-8

### 6.2.2 Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total Proteins | 2,000 |
| Features | 5 (Entry, Length, EC number, Keywords, Gene Ontology) |
| Data Completeness (Entry, Length, Keywords) | 100% |
| Gene Ontology Completeness | 92.2% |
| EC Number Completeness | 57.2% |
| Mean Protein Length | 542 amino acids |
| Median Protein Length | 478 amino acids |
| Length Range | 50 - 1,488 amino acids |
| Standard Deviation (Length) | 308.29 |

### 6.2.3 Feature Descriptions

**Entry**: UniProt accession ID (unique identifier)
- Example: A0A009IHW8, P12345

**Length**: Number of amino acids in the protein sequence
- Integer value
- Range: 50 to 1,488 in our dataset

**EC number**: Enzyme Commission number indicating enzymatic function
- Format: #.#.#.# (e.g., 3.2.1.39)
- Missing for non-enzymes (57.2% have EC numbers)

**Keywords**: Functional and structural annotations
- Semicolon-separated text
- Examples: "3D-structure", "Hydrolase", "Cell wall"
- 100% coverage

**Gene Ontology (GO) terms**: Standardized functional annotations
- Describes molecular function, biological process, cellular component
- Format: "term [GO:ID]"
- 92.2% coverage

## 6.3 Data Preprocessing

### 6.3.1 Data Loading and Validation

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/proteins.tsv', sep='\t')

# Validate structure
assert df.shape[1] == 5, "Expected 5 columns"
assert df['Length'].dtype in [int, 'int64'], "Length must be integer"

# Check for duplicates
duplicates = df['Entry'].duplicated().sum()
print(f"Duplicate entries: {duplicates}")
```

### 6.3.2 Feature Engineering Steps

**Step 1: Target Variable Creation**
```python
# Enzyme classification target
df['has_ec_number'] = df['EC number'].notna()

# Length category target
def categorize_length(length):
    if length < 100: return 'Very_Short'
    elif length < 300: return 'Short'
    elif length < 500: return 'Medium'
    elif length < 1000: return 'Long'
    else: return 'Very_Long'

df['length_category'] = df['Length'].apply(categorize_length)
```

**Step 2: Numerical Features**
```python
import numpy as np

# Length-based features
df['log_length'] = np.log(df['Length'])
df['length_squared'] = df['Length'] ** 2
df['length_percentile'] = df['Length'].rank(pct=True)

# Text-based counts
df['keyword_count'] = df['Keywords'].fillna('').str.count(';') + 1
df['keyword_count'] = df['keyword_count'].where(df['Keywords'].notna(), 0)

df['go_term_count'] = df['Gene Ontology'].fillna('').str.count('\[GO:')
```

**Step 3: Boolean Features**
```python
# Presence indicators
df['has_keywords'] = df['Keywords'].notna()
df['has_go_terms'] = df['Gene Ontology'].notna()

# Flag features
df['flag_very_short'] = (df['Length'] < 100)
df['flag_very_long'] = (df['Length'] > 1000)
df['flag_no_annotation'] = (~df['has_keywords']) & (~df['has_go_terms'])
```

**Step 4: Keyword Category Features**
```python
# Define keyword categories
structural_keywords = ['3d-structure', 'domain', 'repeat', 'membrane']
metabolic_keywords = ['metabolism', 'biosynthesis', 'degradation']
functional_keywords = ['binding', 'transport', 'signaling']

# Create boolean features
def has_keyword_type(keywords, keyword_list):
    if pd.isna(keywords):
        return False
    keywords_lower = keywords.lower()
    return any(kw in keywords_lower for kw in keyword_list)

df['has_structural_keywords'] = df['Keywords'].apply(
    lambda x: has_keyword_type(x, structural_keywords))
df['has_metabolic_keywords'] = df['Keywords'].apply(
    lambda x: has_keyword_type(x, metabolic_keywords))
df['has_functional_keywords'] = df['Keywords'].apply(
    lambda x: has_keyword_type(x, functional_keywords))
```

**Step 5: Derived Features**
```python
# Annotation completeness score
df['basic_annotation_score'] = (
    df['has_keywords'].astype(int) + 
    df['has_go_terms'].astype(int) + 
    df['has_ec_number'].astype(int)
) / 3

# Length category encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['length_category_encoded'] = le.fit_transform(df['length_category'])
```

### 6.3.3 Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Features and target
feature_columns = ['Length', 'log_length', 'keyword_count', 'go_term_count',
                   'has_keywords', 'has_go_terms', 'flag_very_short',
                   'flag_very_long', 'has_structural_keywords',
                   'has_metabolic_keywords', 'basic_annotation_score']

X = df[feature_columns]
y_enzyme = df['has_ec_number']
y_length = df['length_category']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enzyme, test_size=0.2, random_state=42, stratify=y_enzyme)
```

## 6.4 Code Snippets

### 6.4.1 Model Training - Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Initialize model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1  # Use all CPU cores
)

# Train model
rf_model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(
    rf_model, X_train, y_train, 
    cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
```

### 6.4.2 Model Evaluation

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Non-Enzyme', 'Enzyme']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")
```

### 6.4.3 Feature Importance Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), 
            x='importance', y='feature')
plt.title('Top 10 Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/plots/feature_importance.png', dpi=300)
```

---
