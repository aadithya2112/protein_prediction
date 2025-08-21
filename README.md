# 🧬 Protein Prediction Project

A comprehensive analysis project for protein dataset exploration and prediction modeling.

## 📁 Project Structure

```
protein_prediction/
├── 📂 data/                    # Data files
│   ├── proteins.tsv           # Processed dataset (2K proteins)
│   └── proteins_full.tsv      # Full dataset (554K proteins) - Git ignored
├── 📂 docs/                    # Documentation
│   └── data.md               # Data documentation
├── 📂 notebooks/              # Jupyter notebooks for analysis
│   └── protein_analysis_interactive.ipynb  # Interactive EDA notebook
├── 📂 src/                     # Source code
│   └── protein_analysis.py   # Analysis script
├── 📂 outputs/                # Generated outputs
│   ├── 📂 plots/             # Visualizations
│   │   └── protein_analysis_plots.png
│   └── 📂 reports/           # Analysis reports
│       ├── ANALYSIS_REPORT.md
│       └── protein_analysis_summary.csv
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── .venv/                     # Virtual environment (Git ignored)
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if needed)
pip install pandas numpy matplotlib seaborn
```

### 2. Run Analysis

**Interactive Analysis (Recommended):**

```bash
# Open Jupyter notebook
jupyter notebook notebooks/protein_analysis_interactive.ipynb
```

**Script Analysis:**

```bash
# Run Python script
python src/protein_analysis.py
```

## 📊 Dataset Overview

- **Source**: UniProt protein database
- **Size**: 2,000 proteins (sample from 554K full dataset)
- **Features**: Entry ID, Length, EC numbers, Keywords, Gene Ontology terms
- **Quality**: 100% complete for core fields, 92% for GO terms, 57% for EC numbers

## 📈 Analysis Features

### 2️⃣ Dataset Summary

- Basic statistics and data quality assessment
- Missing data analysis
- Column information and data types

### 3️⃣ Statistical Analysis

- Protein length distributions (mean: 542 AA)
- Percentile analysis and outlier detection
- Categorical data summaries

### 4️⃣ Visualizations (6 Types)

1. **Length Distribution** - Histogram with statistical overlays
2. **Box & Violin Plots** - Distribution shape and outliers
3. **Size Categories** - Pie chart of protein size ranges
4. **Data Completeness** - Column completeness assessment
5. **Keyword Analysis** - Most common protein characteristics
6. **Length vs Richness** - Correlation between size and annotation

## 🎯 Key Insights

- **Size Distribution**: Most proteins are 200-800 amino acids
- **Data Quality**: Excellent keyword coverage (100%), strong GO annotation (92%)
- **Protein Categories**: Balanced distribution across size ranges
- **Functional Diversity**: Rich keyword vocabulary with enzyme emphasis

## 📂 File Descriptions

### Data Files

- `data/proteins.tsv` - Working dataset (2K proteins, ~630KB)
- `data/proteins_full.tsv` - Full dataset (554K proteins, ~107MB) [Git ignored]

### Analysis Files

- `notebooks/protein_analysis_interactive.ipynb` - Interactive analysis notebook
- `src/protein_analysis.py` - Standalone analysis script

### Output Files

- `outputs/plots/protein_analysis_plots.png` - Combined visualizations
- `outputs/reports/ANALYSIS_REPORT.md` - Detailed analysis report
- `outputs/reports/protein_analysis_summary.csv` - Key statistics

## 🔧 Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn

## 📝 Notes

- Large dataset files are excluded from Git (see `.gitignore`)
- Virtual environment (`.venv/`) is Git ignored
- All analysis outputs are preserved in `outputs/` directory
- Interactive notebook provides the best exploration experience

## 🚀 Next Steps

- Implement protein classification models
- Add sequence analysis capabilities
- Integrate with protein structure databases
- Develop prediction pipelines

---

_Generated on August 21, 2025_
