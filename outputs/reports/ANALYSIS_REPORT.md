# Protein Dataset Analysis Report

_Generated on August 21, 2025_

## 2. Dataset Summary

### Basic Information

- **Total Rows**: 2,000 proteins
- **Total Columns**: 5 features
- **Dataset Size**: 616KB (trimmed from 107MB full dataset)

### Column Details

| Column        | Data Type | Complete Records | Missing | Completeness |
| ------------- | --------- | ---------------- | ------- | ------------ |
| Entry         | Text      | 2,000            | 0       | 100%         |
| Length        | Integer   | 2,000            | 0       | 100%         |
| EC number     | Text      | 1,144            | 856     | 57.2%        |
| Keywords      | Text      | 2,000            | 0       | 100%         |
| Gene Ontology | Text      | 1,843            | 157     | 92.2%        |

### Sample Data Preview

```
Entry: A0A009IHW8 | Length: 269 | EC: 3.2.2.-; 3.2.2.6
Keywords: 3D-structure;Coiled coil;Hydrolase;NAD
GO Terms: NAD+ nucleosidase activity [GO:0003953]...

Entry: A0A023I7E1 | Length: 796 | EC: 3.2.1.39
Keywords: 3D-structure;Carbohydrate metabolism;Cell wall...
GO Terms: endo-1,3(4)-beta-glucanase activity [GO:0052861]...
```

## 3. Basic Statistics

### Protein Length Distribution

- **Mean**: 542.12 amino acids
- **Median**: 478.00 amino acids
- **Mode**: 283 amino acids
- **Minimum**: 50 amino acids
- **Maximum**: 1,488 amino acids
- **Range**: 1,438 amino acids
- **Standard Deviation**: 308.29
- **25th Percentile**: 322.00
- **75th Percentile**: 718.00

### Data Availability

- **Proteins with EC numbers**: 1,144 (57.2%)
- **Proteins with keywords**: 2,000 (100.0%)
- **Proteins with GO terms**: 1,843 (92.2%)

### Key Insights

1. **Protein sizes vary widely** - from very short (50 AA) to very long (1,488 AA)
2. **Most proteins are medium-sized** - median at 478 amino acids
3. **High data quality** - Keywords available for all proteins, GO terms for 92%
4. **EC numbers less complete** - Only available for 57% of proteins

## 4. Visualizations Created

The analysis generated 6 comprehensive visualizations saved as `protein_analysis_plots.png`:

### Plot 1: Length Distribution Histogram

- Shows the frequency distribution of protein lengths
- Most proteins fall in the 200-800 amino acid range
- Right-skewed distribution with some very long outliers

### Plot 2: Length Box Plot

- Displays quartiles, median, and outliers for protein lengths
- Shows several outliers above 1000 amino acids
- Median around 478 amino acids

### Plot 3: Size Categories Pie Chart

- Categorizes proteins by size ranges:
  - Very Short (<100 AA)
  - Short (100-300 AA)
  - Medium (300-500 AA)
  - Long (500-1000 AA)
  - Very Long (>1000 AA)

### Plot 4: Data Completeness Bar Chart

- Shows percentage of complete data for each column
- Entry and Length: 100% complete
- Keywords: 100% complete
- Gene Ontology: 92.2% complete
- EC number: 57.2% complete

### Plot 5: Top 10 Most Common Keywords

- Horizontal bar chart of most frequently occurring keywords
- Helps identify common protein functions and characteristics
- Shows prevalence of terms like "Hydrolase", "3D-structure", etc.

### Plot 6: Length vs Data Richness Scatter Plot

- Compares protein length against data availability score (0-3)
- Data richness = count of filled fields (EC + Keywords + GO)
- Shows relationship between protein size and annotation completeness

---

## Summary

This dataset provides a good representative sample of protein information with excellent keyword coverage and strong Gene Ontology annotation. The protein length distribution shows typical biological diversity, and the visualizations reveal interesting patterns in protein characteristics and data completeness.
