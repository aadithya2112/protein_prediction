#!/usr/bin/env python3
"""
Protein Dataset Analysis
Basic exploratory data analysis with statistics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def load_data():
    """Load the protein dataset"""
    print("Loading protein dataset...")
    df = pd.read_csv('../data/proteins.tsv', sep='\t')
    print(f"✅ Data loaded successfully!")
    return df


def dataset_summary(df):
    """Display basic dataset information"""
    print("\n" + "="*60)
    print("2. DATASET SUMMARY")
    print("="*60)

    print(f"📊 Number of rows: {len(df):,}")
    print(f"📊 Number of columns: {len(df.columns)}")

    print(f"\n📋 Column Information:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"  {i}. {col}")
        print(f"     - Data type: {dtype}")
        print(f"     - Non-null values: {non_null:,}")
        print(f"     - Missing values: {len(df) - non_null:,}")

    print(f"\n📋 Sample Data (first 5 rows):")
    print("-" * 80)

    # Display sample with better formatting
    sample = df.head()
    for idx, row in sample.iterrows():
        print(f"\nRow {idx + 1}:")
        for col in df.columns:
            value = str(row[col])
            if len(value) > 60:
                value = value[:60] + "..."
            print(f"  {col}: {value}")


def basic_statistics(df):
    """Calculate and display basic statistics"""
    print("\n" + "="*60)
    print("3. BASIC STATISTICS")
    print("="*60)

    # Focus on Length column as it's the main numerical column
    length_col = df['Length']

    print(f"📈 Protein Length Statistics:")
    print(f"   Mean: {length_col.mean():.2f} amino acids")
    print(f"   Median: {length_col.median():.2f} amino acids")
    print(f"   Mode: {length_col.mode().iloc[0]:.0f} amino acids")
    print(f"   Min: {length_col.min():.0f} amino acids")
    print(f"   Max: {length_col.max():.0f} amino acids")
    print(f"   Range: {length_col.max() - length_col.min():.0f} amino acids")
    print(f"   Standard Deviation: {length_col.std():.2f}")
    print(f"   25th Percentile: {length_col.quantile(0.25):.2f}")
    print(f"   75th Percentile: {length_col.quantile(0.75):.2f}")

    # Additional statistics for categorical data
    print(f"\n📊 Categorical Data Summary:")

    # Count entries with EC numbers
    has_ec = df['EC number'].notna().sum()
    print(
        f"   Proteins with EC numbers: {has_ec:,} ({has_ec/len(df)*100:.1f}%)")

    # Count entries with keywords
    has_keywords = df['Keywords'].notna().sum()
    print(
        f"   Proteins with keywords: {has_keywords:,} ({has_keywords/len(df)*100:.1f}%)")

    # Count entries with GO terms
    has_go = df['Gene Ontology (molecular function)'].notna().sum()
    print(f"   Proteins with GO terms: {has_go:,} ({has_go/len(df)*100:.1f}%)")


def create_visualizations(df):
    """Create basic visualizations"""
    print("\n" + "="*60)
    print("4. VISUALIZATIONS")
    print("="*60)

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Histogram of protein lengths
    plt.subplot(2, 3, 1)
    plt.hist(df['Length'], bins=50, alpha=0.7,
             color='skyblue', edgecolor='black')
    plt.title('Distribution of Protein Lengths', fontweight='bold')
    plt.xlabel('Length (amino acids)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # 2. Box plot of protein lengths
    plt.subplot(2, 3, 2)
    plt.boxplot(df['Length'], patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
    plt.title('Protein Length Distribution\n(Box Plot)', fontweight='bold')
    plt.ylabel('Length (amino acids)')
    plt.grid(True, alpha=0.3)

    # 3. Length categories pie chart
    plt.subplot(2, 3, 3)
    length_categories = pd.cut(df['Length'],
                               bins=[0, 100, 300, 500, 1000, float('inf')],
                               labels=['Very Short\n(<100)', 'Short\n(100-300)',
                                       'Medium\n(300-500)', 'Long\n(500-1000)',
                                       'Very Long\n(>1000)'])
    length_counts = length_categories.value_counts()
    plt.pie(length_counts.values, labels=length_counts.index, autopct='%1.1f%%',
            startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
    plt.title('Protein Size Categories', fontweight='bold')

    # 4. Data completeness bar chart
    plt.subplot(2, 3, 4)
    completeness = []
    columns = ['Entry', 'Length', 'EC number',
               'Keywords', 'Gene Ontology (molecular function)']
    for col in columns:
        completeness.append(df[col].notna().sum() / len(df) * 100)

    colors = ['green' if x > 90 else 'orange' if x >
              50 else 'red' for x in completeness]
    bars = plt.bar(range(len(columns)), completeness, color=colors, alpha=0.7)
    plt.title('Data Completeness by Column', fontweight='bold')
    plt.xlabel('Columns')
    plt.ylabel('Completeness (%)')
    plt.xticks(range(len(columns)), [col[:15] + '...' if len(col) > 15 else col
                                     for col in columns], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for bar, pct in zip(bars, completeness):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 5. Top 10 most common keywords
    plt.subplot(2, 3, 5)
    all_keywords = []
    for keywords in df['Keywords'].dropna():
        if pd.notna(keywords):
            all_keywords.extend([k.strip() for k in str(keywords).split(';')])

    keyword_counts = Counter(all_keywords)
    top_keywords = dict(keyword_counts.most_common(10))

    plt.barh(list(top_keywords.keys())[::-1], list(top_keywords.values())[::-1],
             color='coral', alpha=0.7)
    plt.title('Top 10 Most Common Keywords', fontweight='bold')
    plt.xlabel('Frequency')
    plt.grid(True, alpha=0.3)

    # 6. Length vs Data Availability Scatter
    plt.subplot(2, 3, 6)
    # Create a score for data richness (how many fields are filled)
    data_richness = (df['EC number'].notna().astype(int) +
                     df['Keywords'].notna().astype(int) +
                     df['Gene Ontology (molecular function)'].notna().astype(int))

    plt.scatter(df['Length'], data_richness, alpha=0.6, color='purple')
    plt.title('Protein Length vs Data Richness', fontweight='bold')
    plt.xlabel('Length (amino acids)')
    plt.ylabel('Data Richness Score (0-3)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../outputs/plots/protein_analysis_plots.png',
                dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure instead of showing it

    print("✅ Visualizations created and saved as '../outputs/plots/protein_analysis_plots.png'")


def main():
    """Main analysis function"""
    print("🧬 PROTEIN DATASET ANALYSIS")
    print("="*60)

    # Load data
    df = load_data()

    # Run analysis
    dataset_summary(df)
    basic_statistics(df)
    create_visualizations(df)

    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("📊 Check '../outputs/plots/protein_analysis_plots.png' for visualizations")


if __name__ == "__main__":
    main()
